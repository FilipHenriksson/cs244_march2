"""Research agent — drives the research pipeline in two modes.

Default mode:
  A tool-use loop where the LLM freely picks which analyst and reviewer
  tools to call and how many.

Strict mode:
  A code-driven pipeline that always produces the exact same call pattern:
    1. Orchestrator → plan/frame the research          (1 LLM call)
    2. 5 analysts in parallel                          (5 LLM calls)
    3. Orchestrator → draft synthesis from findings     (1 LLM call)
    4. 3 reviewers in parallel on the draft            (3 LLM calls)
    5. Orchestrator → final synthesis with feedback     (1 LLM call)
  Total: 11 LLM calls, 8 tool invocations — deterministic, every time.
"""

import asyncio
import json
from dataclasses import dataclass
from llm import llm_call, register_group, deregister_member
from tools import ALL_TOOL_SCHEMAS, execute_tool, ANALYST_TOOLS, REVIEWER_TOOLS
from prompts import ORCHESTRATOR_SYSTEM_PROMPT

MAX_ROUNDS = 5  # default mode: enough for analysts + reviewers + final

# Fixed tool sets for strict mode
STRICT_ANALYSTS = [
    "analyst_web_research", "analyst_summarizer", "analyst_deep_analysis",
    "analyst_historical_context", "analyst_statistical",
]
STRICT_REVIEWERS = ["review_citations", "review_style", "review_facts"]


@dataclass
class AgentResult:
    """Return value from run_agent: the synthesis text plus call counts."""
    text: str
    llm_calls: int          # total LLM calls (orchestrator + tools)
    tool_calls: int         # total tool invocations (analysts + reviewers)


# ---------------------------------------------------------------------------
# Strict mode — deterministic, code-driven pipeline
# ---------------------------------------------------------------------------

async def _run_strict(prompt: str, session_id: int) -> AgentResult:
    """Run the strict 5-step pipeline with deterministic call counts."""
    sid = f"s{session_id}"
    system = {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}
    messages = [system, {"role": "user", "content": prompt}]

    # Step 1: Orchestrator plans the research
    resp = await llm_call(
        messages=messages, agent_id=f"{sid}:orch",
        call_type="orchestrator", detail="plan",
    )
    messages.append({"role": "assistant", "content": resp.choices[0].message.content})

    # Step 2: Fan out 5 analysts in parallel
    group_id = f"analysts_{sid}"
    register_group(group_id, len(STRICT_ANALYSTS))

    async def _run_analyst(name):
        try:
            return name, await ANALYST_TOOLS[name](
                {"question": prompt}, agent_id=f"{sid}:{name}",
                group_id=group_id,
            )
        finally:
            deregister_member(group_id)

    analyst_results = await asyncio.gather(
        *[_run_analyst(n) for n in STRICT_ANALYSTS])

    analyst_text = "\n\n".join(
        f"--- {name} ---\n{result}" for name, result in analyst_results)
    messages.append({"role": "user", "content":
        f"Here are findings from 5 specialist analysts:\n\n{analyst_text}\n\n"
        f"Write a comprehensive draft synthesis of these findings."
    })

    # Step 3: Orchestrator drafts synthesis
    resp = await llm_call(
        messages=messages, agent_id=f"{sid}:orch",
        call_type="orchestrator", detail="synthesize",
    )
    draft = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": draft})

    # Step 4: Fan out 3 reviewers in parallel
    group_id = f"reviewers_{sid}"
    register_group(group_id, len(STRICT_REVIEWERS))

    async def _run_reviewer(name):
        try:
            return name, await REVIEWER_TOOLS[name](
                {"text": draft}, agent_id=f"{sid}:{name}",
                group_id=group_id,
            )
        finally:
            deregister_member(group_id)

    reviewer_results = await asyncio.gather(
        *[_run_reviewer(n) for n in STRICT_REVIEWERS])

    reviewer_text = "\n\n".join(
        f"--- {name} ---\n{result}" for name, result in reviewer_results)
    messages.append({"role": "user", "content":
        f"Here is feedback from 3 reviewers:\n\n{reviewer_text}\n\n"
        f"Incorporate the feedback and produce your final polished synthesis."
    })

    # Step 5: Orchestrator final synthesis
    resp = await llm_call(
        messages=messages, agent_id=f"{sid}:orch",
        call_type="orchestrator", detail="final",
    )

    n_analysts = len(STRICT_ANALYSTS)
    n_reviewers = len(STRICT_REVIEWERS)
    return AgentResult(
        text=resp.choices[0].message.content,
        llm_calls=3 + n_analysts + n_reviewers,   # 11
        tool_calls=n_analysts + n_reviewers,       # 8
    )


# ---------------------------------------------------------------------------
# Default mode — LLM-driven tool-use loop
# ---------------------------------------------------------------------------

async def _fan_out_tool_calls(tool_calls, session_id: int,
                              round_num: int) -> list[dict]:
    """Execute a batch of tool_calls in parallel as a scheduler group."""
    group_id = f"round{round_num}_s{session_id}"
    register_group(group_id, len(tool_calls))

    async def _run_one(tc):
        try:
            args = json.loads(tc.function.arguments)
            return await execute_tool(
                tc.function.name, args,
                agent_id=f"s{session_id}:{tc.function.name}",
                group_id=group_id,
            )
        finally:
            deregister_member(group_id)

    results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
    return [
        {"role": "tool", "tool_call_id": tc.id, "content": result}
        for tc, result in zip(tool_calls, results)
    ]


async def _run_default(prompt: str, session_id: int) -> AgentResult:
    """Run the default tool-use loop where the LLM picks tools freely."""
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    orch_calls = 0
    total_tool_calls = 0

    for round_num in range(MAX_ROUNDS):
        resp = await llm_call(
            messages=messages,
            agent_id=f"s{session_id}:orch",
            call_type="orchestrator",
            detail=f"round-{round_num}",
            tools=ALL_TOOL_SCHEMAS,
        )
        orch_calls += 1
        msg = resp.choices[0].message

        if msg.tool_calls:
            total_tool_calls += len(msg.tool_calls)
            messages.append(msg)
            tool_results = await _fan_out_tool_calls(
                msg.tool_calls, session_id, round_num,
            )
            messages.extend(tool_results)
            continue

        return AgentResult(
            text=msg.content,
            llm_calls=orch_calls + total_tool_calls,
            tool_calls=total_tool_calls,
        )

    # Exhausted rounds — force a final text-only call
    resp = await llm_call(
        messages=messages,
        agent_id=f"s{session_id}:orch",
        call_type="orchestrator",
        detail="final",
    )
    orch_calls += 1
    return AgentResult(
        text=resp.choices[0].message.content,
        llm_calls=orch_calls + total_tool_calls,
        tool_calls=total_tool_calls,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(prompt: str, session_id: int = 0,
                    prompt_mode: str = "default") -> AgentResult:
    """Run one research session.

    Parameters
    ----------
    prompt : str
        The research topic.
    session_id : int
        Session identifier (used for tracing and group IDs).
    prompt_mode : str
        "default" — LLM freely picks tools.
        "strict"  — code-driven pipeline, deterministic call counts.
    """
    if prompt_mode == "strict":
        return await _run_strict(prompt, session_id)
    return await _run_default(prompt, session_id)
