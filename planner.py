"""
Planner layer: calls a local Ollama model to emit and validate a TaskPlan DAG.

Design notes
------------
- Works for any task domain (coding benchmarks, research, QA, etc.); the system
  prompt is intentionally domain-agnostic.
- Planner calls go directly to the local Ollama endpoint and intentionally bypass
  the shared remote-model scheduler and rate limiter.  This prevents local calls
  from consuming shared RPM/TPM budget and distorting scheduler fairness for the
  executor agents.
- Tracing is still recorded with call_type="planner_plan" so the plan phase is
  visible in trace summaries alongside all other call types.
- On a parse/validation failure the error is fed back to the model as a
  correction message (up to PLANNER_MAX_RETRIES total attempts).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from openai import AsyncOpenAI

from task_types import TaskPlan
from trace import trace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------

PLANNER_MODEL: str = os.getenv("PLANNER_MODEL", "llama3")
PLANNER_BASE_URL: str = os.getenv("PLANNER_BASE_URL", "http://localhost:11434/v1")
# Ollama's OpenAI-compatible endpoint does not enforce an API key, but the
# openai library requires a non-empty value.
PLANNER_API_KEY: str = os.getenv("PLANNER_API_KEY", "ollama")
PLANNER_MAX_RETRIES: int = int(os.getenv("PLANNER_MAX_RETRIES", "3"))
PLANNER_TEMPERATURE: float = float(os.getenv("PLANNER_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a task planner. Given a goal or problem, decompose it into a DAG of \
explicit subtasks that together solve the problem completely.

The goal may be anything: a coding problem, a benchmark question, a research \
topic, a data-processing pipeline, or any other task. Decompose it accordingly.

Return ONLY valid JSON with this exact schema (no markdown, no explanation):

{
  "goal": "<the original goal or problem statement>",
  "tasks": [
    {
      "id": "<short snake_case identifier, e.g. step_1>",
      "title": "<one-line human-readable name>",
      "instructions": "<full, self-contained instructions for the executor>",
      "expected_output": "<description of what a correct result looks like>",
      "depends_on": ["<id of a prerequisite task>"],
      "inputs": ["<id of a task whose output to forward as context>"],
      "executor_type": "agent",
      "agent_hint": null,
      "metadata": {}
    }
  ],
  "final_task_id": "<id of the task whose output is the final answer>",
  "output_order": []
}

Rules:
- depends_on lists every task that must finish before this one starts.
- inputs is a subset of (or equal to) depends_on; list tasks whose output
  the executor needs as explicit context.
- There must be no dependency cycles.
- Use executor_type "agent" for tasks that require tool use or multi-step
  reasoning (e.g. code execution, retrieval, exploration); use "llm" for
  tasks that only need a single synthesis or summarization call.
- final_task_id must be the id of a task in the tasks list.
- Return ONLY the raw JSON object — no markdown fences, no prose, no comments.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """
    Extract the JSON object from model output.

    Handles the common case where an instruction-tuned model wraps its answer
    in markdown code fences despite being told not to.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        return fence_match.group(1).strip()
    # Fall back: find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _make_client() -> AsyncOpenAI:
    """Return a fresh AsyncOpenAI client pointed at the local Ollama endpoint."""
    return AsyncOpenAI(
        base_url=PLANNER_BASE_URL,
        api_key=PLANNER_API_KEY,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid TaskPlan after all retries."""


async def plan(
    goal: str,
    session_context: str = "",
    agent_id: str = "planner",
    session_id: int = 0,
) -> TaskPlan:
    """
    Call the local planner model to produce a validated TaskPlan DAG.

    Parameters
    ----------
    goal:
        The task or problem to plan.  Can be a coding problem, benchmark
        question, research topic, or any other goal.
    session_context:
        Optional additional context to include in the planner prompt (e.g.
        a problem description, prior conversation history, or constraints).
    agent_id:
        Short identifier used in trace records.  Defaults to "planner".
    session_id:
        Session index used to build a fully-qualified trace agent id in the
        format ``s{session_id}:{agent_id}``.

    Returns
    -------
    A validated :class:`TaskPlan` instance whose dependency graph is
    cycle-free and internally consistent.

    Raises
    ------
    PlannerError
        If the model fails to produce a valid plan after all retries.
    """
    client = _make_client()
    full_agent_id = f"s{session_id}:{agent_id}"

    user_content = goal
    if session_context:
        user_content = f"{goal}\n\nAdditional context:\n{session_context}"

    messages: list[dict] = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error: Exception | None = None
    last_raw: str = ""

    for attempt in range(1, PLANNER_MAX_RETRIES + 1):
        trace_call = trace.start_call(
            agent_id=full_agent_id,
            call_type="planner_plan",
            detail=f"attempt-{attempt}",
        )
        t_start = time.time()
        try:
            response = await client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=messages,
                temperature=PLANNER_TEMPERATURE,
            )
            last_raw = response.choices[0].message.content or ""
            trace.end_call(trace_call)

            json_text = _extract_json(last_raw)
            task_plan = TaskPlan.from_json(json_text)
            logger.info(
                "Planner succeeded on attempt %d/%d — %d tasks in plan",
                attempt,
                PLANNER_MAX_RETRIES,
                len(task_plan.tasks),
            )
            return task_plan

        except Exception as exc:
            last_error = exc
            # Ensure trace records the end even on failure
            if trace_call.end == 0.0:
                trace.end_call(trace_call)

            logger.warning(
                "Planner attempt %d/%d failed: %s",
                attempt,
                PLANNER_MAX_RETRIES,
                exc,
            )

            if attempt < PLANNER_MAX_RETRIES:
                # Feed the error back to give the model a chance to self-correct
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as a valid "
                            "TaskPlan.\n"
                            f"Error: {exc}\n\n"
                            "Please return ONLY valid JSON matching the schema above, "
                            "with no markdown fences or extra text."
                        ),
                    }
                )

    raise PlannerError(
        f"Planner failed after {PLANNER_MAX_RETRIES} attempt(s). "
        f"Last error: {last_error}"
    ) from last_error
