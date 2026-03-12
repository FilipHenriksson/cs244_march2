"""Orchestration pipeline: breakdown -> agents -> synthesis -> reviewers -> finalize.

Each orchestrate() call runs one research session through five stages:

  1. Breakdown  — LLM splits the topic into NUM_AGENTS sub-questions
  2. Research agent fan-out    — parallel research agents investigate each sub-question (tool-use loop)
  3. Synthesis  — LLM combines agent findings into a coherent draft
  4. Fan-out    — parallel reviewers (citations, style, facts) critique the draft
  5. Finalize   — LLM revises the draft incorporating reviewer feedback

All LLM calls route through llm.llm_call(), which delegates to the active
scheduler.  Fan-out stages register a "group" with the scheduler so that
group-aware policies (e.g. MapReduce) can prioritize calls whose sibling
tasks are nearest completion.
"""

import asyncio
from llm import llm_call, register_group, deregister_member
from agent import run_agent
from stages import (
    NUM_AGENTS, BreakdownStage, SynthesisStage, FinalizeStage, REVIEWERS,
)


def _parse_sub_questions(text: str) -> list[str]:
    """Extract up to NUM_AGENTS sub-questions from numbered LLM output."""
    questions = []
    for line in text.strip().split("\n"):
        cleaned = line.strip().lstrip("0123456789.)- ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:NUM_AGENTS]


async def _run_stage(stage, session_id: int, group_id: str = None,
                     **kwargs) -> str:
    """Execute a single WorkflowStage and return the LLM's text response."""
    resp = await llm_call(
        messages=stage.message_factory(**kwargs),
        agent_id=f"s{session_id}:{stage.id_suffix}",
        call_type=stage.call_type,
        group_id=group_id,
    )
    return resp.choices[0].message.content


async def _fan_out(group_id: str, coros: list) -> list:
    """Execute coroutines as a scheduler group, deregistering each on completion.

    Group-aware schedulers (e.g. MapReduce) use the group membership count to
    boost priority for calls whose fan-out peers are already done.  Each
    coroutine's completion triggers deregister_member(), updating that count.
    """
    register_group(group_id, len(coros))

    async def _with_cleanup(coro):
        try:
            return await coro
        finally:
            deregister_member(group_id)

    return list(await asyncio.gather(*[_with_cleanup(c) for c in coros]))


async def orchestrate(prompt: str, session_id: int = 0) -> str:
    """Run the full five-stage research pipeline for one session."""

    # Stage 1: Breakdown — split topic into sub-questions
    sub_q_text = await _run_stage(BreakdownStage(), session_id, prompt=prompt)
    sub_questions = _parse_sub_questions(sub_q_text)

    # Stage 2: Fan-out — parallel research agents
    agent_group = f"agents_s{session_id}"
    results = await _fan_out(agent_group, [
        run_agent(q, i, session_id, group_id=agent_group)
        for i, q in enumerate(sub_questions)
    ])

    # Stage 3: Fan-in — synthesize findings
    findings = "\n\n".join(
        f"### Agent {i} — {sub_questions[i]}\n{result}"
        for i, result in enumerate(results)
    )
    synthesis_text = await _run_stage(
        SynthesisStage(), session_id, prompt=prompt, findings=findings,
    )

    # Stage 4: Fan-out — parallel reviewers
    review_group = f"reviewers_s{session_id}"
    review_results = await _fan_out(review_group, [
        _run_stage(stage, session_id, group_id=review_group,
                   prompt=prompt, synthesis_text=synthesis_text)
        for stage in REVIEWERS
    ])

    # Stage 5: Fan-in — finalize with reviewer feedback
    review_feedback = "\n\n".join(
        f"### {stage.call_type} Review\n{result}"
        for stage, result in zip(REVIEWERS, review_results)
    )
    return await _run_stage(
        FinalizeStage(), session_id,
        prompt=prompt, synthesis_text=synthesis_text,
        review_feedback=review_feedback,
    )
