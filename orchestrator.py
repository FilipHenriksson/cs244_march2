"""Orchestration pipeline: breakdown -> agents -> synthesis -> reviewers -> finalize.

Each orchestrate() call runs one research session through five stages:

  1. Breakdown  — LLM splits the topic into NUM_AGENTS sub-questions
  2. Fan-out    — parallel research agents investigate each sub-question (tool-use loop)
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
from prompts import REVIEWER_CONFIGS

NUM_AGENTS = 5


def _parse_sub_questions(text: str) -> list[str]:
    """Extract up to NUM_AGENTS sub-questions from numbered LLM output."""
    questions = []
    for line in text.strip().split("\n"):
        cleaned = line.strip().lstrip("0123456789.)- ").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:NUM_AGENTS]


async def _run_reviewer(synthesis_text: str, prompt: str, config: dict,
                        session_id: int, group_id: str = None) -> str:
    """Single review pass — one LLM call, no tools."""
    resp = await llm_call(
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {
                "role": "user",
                "content": (
                    f"Original research topic: {prompt}\n\n"
                    f"Synthesis to review:\n{synthesis_text}"
                ),
            },
        ],
        agent_id=f"s{session_id}:{config['id_suffix']}",
        call_type=config["call_type"],
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def orchestrate(prompt: str, session_id: int = 0) -> str:
    """Run the full five-stage research pipeline for one session."""
    sid = f"s{session_id}"

    # Stage 1: Breakdown — split topic into sub-questions
    breakdown_resp = await llm_call(
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a research coordinator. Given a research topic, generate exactly "
                    f"{NUM_AGENTS} distinct sub-questions that together cover the topic comprehensively. "
                    f"Return ONLY the questions, one per line, numbered 1-{NUM_AGENTS}."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        agent_id=f"{sid}:orch",
        call_type="breakdown",
    )
    sub_questions = _parse_sub_questions(breakdown_resp.choices[0].message.content)

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
    synthesis_resp = await llm_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research synthesizer. Combine the findings from multiple "
                    "research agents into a coherent, well-structured summary. Highlight "
                    "key insights and connections between findings."
                ),
            },
            {
                "role": "user",
                "content": f"Original topic: {prompt}\n\nFindings:\n{findings}",
            },
        ],
        agent_id=f"{sid}:orch",
        call_type="synthesis",
    )
    synthesis_text = synthesis_resp.choices[0].message.content

    # Stage 4: Fan-out — parallel reviewers
    review_group = f"reviewers_s{session_id}"
    review_results = await _fan_out(review_group, [
        _run_reviewer(synthesis_text, prompt, cfg, session_id, group_id=review_group)
        for cfg in REVIEWER_CONFIGS
    ])

    # Stage 5: Fan-in — finalize with reviewer feedback
    review_feedback = "\n\n".join(
        f"### {cfg['call_type'].replace('review_', '').replace('_', ' ').title()} Review\n{result}"
        for cfg, result in zip(REVIEWER_CONFIGS, review_results)
    )
    finalize_resp = await llm_call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research editor producing the final version of a research "
                    "synthesis. You have received the original synthesis and feedback from "
                    "three reviewers (citation checker, style/grammar checker, and fact "
                    "checker). Your job is to:\n"
                    "1. Address all valid reviewer feedback by revising the synthesis.\n"
                    "2. Fix citation gaps by adding source attributions where flagged.\n"
                    "3. Fix grammar and style issues identified by the style reviewer.\n"
                    "4. Remove or qualify any factual claims flagged as incorrect or "
                    "questionable.\n"
                    "5. Preserve the overall structure and key insights of the original.\n\n"
                    "Return ONLY the final polished synthesis — do not include reviewer "
                    "comments or meta-commentary about what you changed."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original topic: {prompt}\n\n"
                    f"Original synthesis:\n{synthesis_text}\n\n"
                    f"Reviewer feedback:\n{review_feedback}"
                ),
            },
        ],
        agent_id=f"{sid}:orch",
        call_type="finalize",
    )
    return finalize_resp.choices[0].message.content
