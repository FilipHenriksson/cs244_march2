import asyncio
from llm import llm_call
from agent import run_agent

NUM_AGENTS = 5


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
    results = await asyncio.gather(
        *[run_agent(q, i, session_id) for i, q in enumerate(sub_questions)]
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

    return synthesis_resp.choices[0].message.content
