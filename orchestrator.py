
import asyncio
import llm as llm_mod
from llm import llm_call
from agent import run_agent

NUM_AGENTS = 5

REVIEWER_CONFIGS = [
    {
        "id_suffix": "r0",
        "call_type": "review_citations",
        "system_prompt": (
            "You are a citation and source checker. You will receive a research "
            "synthesis document. Your job is to:\n"
            "1. Identify every factual claim in the text.\n"
            "2. Check whether each claim is attributed to a source or left unsourced.\n"
            "3. Flag any claims that lack citations or have vague attribution "
            "(e.g., 'studies show' without specifics).\n"
            "4. Suggest where additional citations would strengthen the text.\n\n"
            "Return your findings as a structured list with the format:\n"
            "- CLAIM: <the claim>\n"
            "  STATUS: [cited | unsourced | vague]\n"
            "  SUGGESTION: <optional improvement>\n\n"
            "End with a brief overall assessment of citation quality."
        ),
    },
    {
        "id_suffix": "r1",
        "call_type": "review_style",
        "system_prompt": (
            "You are a style and grammar editor. You will receive a research "
            "synthesis document. Your job is to:\n"
            "1. Identify grammar errors, awkward phrasing, or unclear sentences.\n"
            "2. Check for consistent tone (formal academic style).\n"
            "3. Flag any structural issues (poor transitions, missing topic sentences, "
            "logical flow problems).\n"
            "4. Suggest specific rewrites for any problematic passages.\n\n"
            "Return your findings as a structured list with the format:\n"
            "- ISSUE: <description of the problem>\n"
            "  LOCATION: <quote the relevant text>\n"
            "  SUGGESTION: <specific fix>\n\n"
            "End with a brief overall assessment of writing quality."
        ),
    },
    {
        "id_suffix": "r2",
        "call_type": "review_facts",
        "system_prompt": (
            "You are a fact checker. You will receive a research synthesis document. "
            "Your job is to:\n"
            "1. Identify every factual claim (statistics, dates, names, causal claims).\n"
            "2. Assess whether each claim is plausible and internally consistent with "
            "the rest of the document.\n"
            "3. Flag any claims that appear incorrect, outdated, exaggerated, or "
            "contradicted by other parts of the text.\n"
            "4. Note any claims you cannot verify that should be double-checked.\n\n"
            "Return your findings as a structured list with the format:\n"
            "- CLAIM: <the claim>\n"
            "  VERDICT: [plausible | questionable | likely incorrect | unverifiable]\n"
            "  REASONING: <brief explanation>\n\n"
            "End with a brief overall assessment of factual accuracy."
        ),
    },
]


async def run_reviewer(synthesis_text: str, prompt: str, config: dict,
                       session_id: int, group_id: str = None) -> str:
    """Run a single review agent (no tools, single LLM call)."""
    sid = f"s{session_id}"
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
        agent_id=f"{sid}:{config['id_suffix']}",
        call_type=config["call_type"],
        group_id=group_id,
    )
    return resp.choices[0].message.content


async def orchestrate(prompt: str, session_id: int = 0) -> str:
    """Fan out to research agents, fan in to synthesize."""
    sid = f"s{session_id}"

    # Step 1: Break topic into sub-questions
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
    sub_questions_text = breakdown_resp.choices[0].message.content

    # Parse sub-questions
    lines = [l.strip() for l in sub_questions_text.strip().split("\n") if l.strip()]
    sub_questions = []
    for line in lines:
        cleaned = line.lstrip("0123456789.)- ").strip()
        if cleaned:
            sub_questions.append(cleaned)
    sub_questions = sub_questions[:NUM_AGENTS]

    # Step 2: Fan out
    agent_group = f"agents_s{session_id}"
    llm_mod._scheduler.register_group(agent_group, len(sub_questions))

    async def _run_agent_wrapped(q, i, sid, gid):
        try:
            return await run_agent(q, i, sid, group_id=gid)
        finally:
            llm_mod._scheduler.deregister_member(gid)

    results = await asyncio.gather(
        *[_run_agent_wrapped(q, i, session_id, agent_group)
          for i, q in enumerate(sub_questions)]
    )

    # Step 3: Fan in — synthesize
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

    # Step 4: Fan out — review
    review_group = f"reviewers_s{session_id}"
    llm_mod._scheduler.register_group(review_group, len(REVIEWER_CONFIGS))

    async def _run_reviewer_wrapped(synth, p, cfg, sid, gid):
        try:
            return await run_reviewer(synth, p, cfg, sid, group_id=gid)
        finally:
            llm_mod._scheduler.deregister_member(gid)

    review_results = await asyncio.gather(
        *[
            _run_reviewer_wrapped(synthesis_text, prompt, config, session_id, review_group)
            for config in REVIEWER_CONFIGS
        ]
    )

    # Step 5: Fan in — finalize
    review_feedback = "\n\n".join(
        f"### {config['call_type'].replace('review_', '').replace('_', ' ').title()} Review\n{result}"
        for config, result in zip(REVIEWER_CONFIGS, review_results)
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
