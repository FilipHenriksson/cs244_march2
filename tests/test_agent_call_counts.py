"""Tests that the agent produces the expected number of LLM calls.

Strict mode uses a code-driven pipeline:
  1. Orchestrator → plan          (1 LLM call)
  2. 5 analysts in parallel       (5 LLM calls)
  3. Orchestrator → synthesize    (1 LLM call)
  4. 3 reviewers in parallel      (3 LLM calls)
  5. Orchestrator → final         (1 LLM call)
  = 11 total LLM calls, 8 tool invocations

Default mode uses a tool-use loop where the LLM picks tools freely.
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_agent, AgentResult


def _make_tool_call(name, arguments='{"question":"test"}', call_id=None):
    tc = SimpleNamespace()
    tc.id = call_id or f"call_{name}"
    tc.type = "function"
    tc.function = SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_response(content=None, tool_calls=None):
    msg = SimpleNamespace()
    msg.content = content or ""
    msg.tool_calls = tool_calls
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Strict mode tests
# ---------------------------------------------------------------------------

def test_strict_mode_call_counts():
    """Strict mode: 3 orch + 5 analyst + 3 reviewer = 11 LLM calls, 8 tool calls."""
    call_log = []

    async def mock_llm_call(messages, session_id, call_key, label="",
                            **kwargs):
        call_log.append((session_id, call_key, label))
        return _make_response(content=f"Response from {call_key}")

    async def _run():
        call_log.clear()
        with patch("agent.llm_call", side_effect=mock_llm_call), \
             patch("agent.ANALYST_TOOLS", {
                 name: AsyncMock(return_value=f"Result from {name}")
                 for name in ["analyst_web_research", "analyst_summarizer",
                              "analyst_deep_analysis",
                              "analyst_historical_context",
                              "analyst_statistical"]
             }), \
             patch("agent.REVIEWER_TOOLS", {
                 name: AsyncMock(return_value=f"Result from {name}")
                 for name in ["review_citations", "review_style",
                              "review_facts"]
             }):
            return await run_agent("Test topic", session_id=0,
                                   prompt_mode="strict")

    result = asyncio.run(_run())

    assert isinstance(result, AgentResult)

    # Verify exact call pattern
    orch_calls = [c for c in call_log if c[1].startswith("orchestrator:")]
    assert len(orch_calls) == 3, f"Expected 3 orch calls, got {len(orch_calls)}"
    assert orch_calls[0][1] == "orchestrator:plan"
    assert orch_calls[1][1] == "orchestrator:synthesize"
    assert orch_calls[2][1] == "orchestrator:final"

    # Verify totals
    assert result.llm_calls == 11, f"Expected 11 llm_calls, got {result.llm_calls}"
    assert result.tool_calls == 8, f"Expected 8 tool_calls, got {result.tool_calls}"


def test_strict_mode_calls_correct_tools():
    """Strict mode calls exactly the 5 hardcoded analysts and 3 reviewers."""
    called_analysts = []
    called_reviewers = []

    async def mock_llm_call(messages, session_id, call_key, label="",
                            **kwargs):
        return _make_response(content="text")

    def make_mock_analyst(name):
        async def fn(arguments, session_id):
            called_analysts.append(name)
            return f"Result from {name}"
        return fn

    def make_mock_reviewer(name):
        async def fn(arguments, session_id):
            called_reviewers.append(name)
            return f"Result from {name}"
        return fn

    async def _run():
        called_analysts.clear()
        called_reviewers.clear()
        with patch("agent.llm_call", side_effect=mock_llm_call), \
             patch("agent.ANALYST_TOOLS", {
                 n: make_mock_analyst(n) for n in [
                     "analyst_web_research", "analyst_summarizer",
                     "analyst_deep_analysis", "analyst_historical_context",
                     "analyst_statistical"]
             }), \
             patch("agent.REVIEWER_TOOLS", {
                 n: make_mock_reviewer(n) for n in [
                     "review_citations", "review_style", "review_facts"]
             }):
            return await run_agent("Test topic", session_id=0,
                                   prompt_mode="strict")

    asyncio.run(_run())

    assert sorted(called_analysts) == sorted([
        "analyst_web_research", "analyst_summarizer", "analyst_deep_analysis",
        "analyst_historical_context", "analyst_statistical",
    ]), f"Wrong analysts called: {called_analysts}"

    assert sorted(called_reviewers) == sorted([
        "review_citations", "review_style", "review_facts",
    ]), f"Wrong reviewers called: {called_reviewers}"


def test_strict_mode_passes_draft_to_reviewers():
    """Strict mode passes the orchestrator's draft synthesis to each reviewer."""
    reviewer_inputs = []

    async def mock_llm_call(messages, session_id, call_key, label="",
                            **kwargs):
        if call_key == "orchestrator:synthesize":
            return _make_response(content="THE DRAFT SYNTHESIS")
        return _make_response(content="text")

    def make_mock_reviewer(name):
        async def fn(arguments, session_id):
            reviewer_inputs.append((name, arguments.get("text")))
            return "review feedback"
        return fn

    async def _run():
        reviewer_inputs.clear()
        with patch("agent.llm_call", side_effect=mock_llm_call), \
             patch("agent.ANALYST_TOOLS", {
                 n: AsyncMock(return_value="analyst result")
                 for n in ["analyst_web_research", "analyst_summarizer",
                            "analyst_deep_analysis",
                            "analyst_historical_context",
                            "analyst_statistical"]
             }), \
             patch("agent.REVIEWER_TOOLS", {
                 n: make_mock_reviewer(n) for n in [
                     "review_citations", "review_style", "review_facts"]
             }):
            return await run_agent("Test topic", session_id=0,
                                   prompt_mode="strict")

    asyncio.run(_run())

    for name, text in reviewer_inputs:
        assert text == "THE DRAFT SYNTHESIS", \
            f"{name} got wrong text: {text!r}"


# ---------------------------------------------------------------------------
# Default mode tests
# ---------------------------------------------------------------------------

ANALYST_CALLS = [
    _make_tool_call("analyst_web_research"),
    _make_tool_call("analyst_summarizer"),
    _make_tool_call("analyst_deep_analysis"),
]

REVIEWER_CALLS = [
    _make_tool_call("review_citations", '{"text":"draft"}'),
    _make_tool_call("review_style", '{"text":"draft"}'),
]


def test_default_mode_tracks_counts():
    """Default mode: LLM picks tools freely, verify counts are tracked."""

    async def mock_llm_call(messages, session_id, call_key, label="",
                            **kwargs):
        if call_key.startswith("orchestrator:"):
            if "round-0" in call_key:
                return _make_response(tool_calls=ANALYST_CALLS)
            elif "round-1" in call_key:
                return _make_response(tool_calls=REVIEWER_CALLS)
            else:
                return _make_response(content="Final.")
        return _make_response(content="tool result")

    async def mock_execute_tool(name, arguments, session_id):
        return "tool result"

    async def _run():
        with patch("agent.llm_call", side_effect=mock_llm_call), \
             patch("agent.execute_tool", side_effect=mock_execute_tool):
            return await run_agent("Test topic", session_id=0,
                                   prompt_mode="default")

    result = asyncio.run(_run())
    assert isinstance(result, AgentResult)
    # 3 orch + 3 analyst + 2 reviewer = 8 LLM calls
    assert result.llm_calls == 8, f"Expected 8, got {result.llm_calls}"
    assert result.tool_calls == 5, f"Expected 5, got {result.tool_calls}"


def test_default_mode_immediate_answer():
    """If LLM returns text immediately, 1 LLM call, 0 tool calls."""

    async def mock_llm_call(messages, session_id, call_key, label="",
                            **kwargs):
        return _make_response(content="Immediate answer.")

    async def _run():
        with patch("agent.llm_call", side_effect=mock_llm_call), \
             patch("agent.execute_tool"):
            return await run_agent("Test topic", session_id=0)

    result = asyncio.run(_run())
    assert result.llm_calls == 1
    assert result.tool_calls == 0


if __name__ == "__main__":
    tests = [
        test_strict_mode_call_counts,
        test_strict_mode_calls_correct_tools,
        test_strict_mode_passes_draft_to_reviewers,
        test_default_mode_tracks_counts,
        test_default_mode_immediate_answer,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed!")
