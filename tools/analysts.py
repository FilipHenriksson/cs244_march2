"""Nine analyst tools — each investigates a question from a specialized angle.

Every analyst is a single LLM call with a focused system prompt.  The prompts
are deliberately varied in scope and expected output length so that different
analysts produce responses of meaningfully different sizes — some are quick
(bullet-point summaries, stats tables) and others are deep (multi-page analysis,
historical timelines).  This variation exercises scheduler prioritization of
heterogeneous workloads.
"""

from llm import llm_call


# ---------------------------------------------------------------------------
# Analyst definitions: (name, description, system_prompt)
#
# Ordered roughly from SHORT expected output to LONG expected output.
# ---------------------------------------------------------------------------

_ANALYSTS = [
    (
        "analyst_statistical",
        "Focus on quantitative data, metrics, and numerical evidence.",
        "You are a statistical analyst. Your response must be SHORT and "
        "data-dense. Present ONLY a numbered list of the most important "
        "statistics, data points, and quantitative comparisons related to "
        "the question. Each item should be one line: the metric, its value, "
        "and the source. No prose, no narrative — just the numbers. "
        "Aim for 8-12 bullet points maximum.",
    ),
    (
        "analyst_summarizer",
        "Distill key points about a question into a concise, organized summary.",
        "You are a research summarizer. Produce a BRIEF executive summary "
        "of the topic in no more than 5-7 bullet points. Each bullet should "
        "be a single sentence capturing one key takeaway. Do not elaborate "
        "or provide background — assume the reader is an expert who just "
        "needs the headlines. Keep total length under 200 words.",
    ),
    (
        "analyst_counterarguments",
        "Examine the question critically — identify counterarguments, limitations, and risks.",
        "You are a critical analysis specialist. Identify the 3-5 strongest "
        "counterarguments, limitations, or risks related to the topic. For "
        "each one, state the counterargument in one sentence and provide "
        "a brief (2-3 sentence) explanation of why it matters. Be specific "
        "and cite concrete examples where possible. Keep it focused and concise.",
    ),
    (
        "analyst_web_research",
        "Search for and report the most relevant, up-to-date information on a question.",
        "You are a web research specialist. Produce a structured research "
        "briefing with the following sections:\n"
        "1. KEY FINDINGS (5-8 major findings with specific facts and figures)\n"
        "2. NOTABLE SOURCES (list the most authoritative organizations, "
        "papers, or institutions working on this topic)\n"
        "3. RECENT DEVELOPMENTS (what happened in the last 1-2 years)\n\n"
        "Be specific — include names, dates, numbers, and institutional "
        "affiliations. Moderate length: thorough but not exhaustive.",
    ),
    (
        "analyst_comparative",
        "Compare and contrast different approaches, alternatives, or perspectives.",
        "You are a comparative analyst. Produce a detailed comparison of "
        "the major approaches, methods, or schools of thought related to "
        "this question. Structure your response as:\n"
        "1. Identify 3-5 distinct approaches or alternatives\n"
        "2. For each: describe the approach, its key strengths, its key "
        "weaknesses, and which contexts it works best in\n"
        "3. Provide a summary comparison table (text-based) rating each "
        "approach on relevant dimensions\n"
        "4. Give your assessment of which approach is most promising and why\n\n"
        "Be thorough — this is a reference document for decision-makers.",
    ),
    (
        "analyst_trend_forecast",
        "Identify current trends, emerging patterns, and likely future developments.",
        "You are a trend and forecasting analyst. Produce a forward-looking "
        "analysis with:\n"
        "1. CURRENT STATE: Where things stand right now (brief)\n"
        "2. SHORT-TERM TRENDS (1-3 years): What is actively changing\n"
        "3. MEDIUM-TERM OUTLOOK (3-10 years): Evidence-based projections\n"
        "4. LONG-TERM POSSIBILITIES (10+ years): Speculative but grounded scenarios\n"
        "5. KEY UNCERTAINTIES: What could change these projections\n"
        "6. SIGNALS TO WATCH: Specific indicators that would confirm or "
        "disconfirm your projections\n\n"
        "Support each projection with evidence. Include specific timelines "
        "and milestones where possible.",
    ),
    (
        "analyst_deep_analysis",
        "Provide a thorough examination of underlying mechanisms and causal relationships.",
        "You are a deep analysis specialist. Provide an EXHAUSTIVE, detailed "
        "examination of the given question. Your analysis must cover:\n"
        "1. MECHANISM: How does this work at a fundamental level? Explain the "
        "underlying science, engineering, or theory in detail.\n"
        "2. CAUSAL CHAIN: What causes what? Map out the causal relationships "
        "and feedback loops.\n"
        "3. DEPENDENCIES: What does this depend on? What are the prerequisites "
        "and enabling conditions?\n"
        "4. SECOND-ORDER EFFECTS: What are the non-obvious consequences and "
        "ripple effects?\n"
        "5. EDGE CASES: Where does the standard understanding break down?\n\n"
        "Go deep. This should be the most thorough analysis possible — imagine "
        "you are writing a technical reference that experts will consult. "
        "Use precise terminology and explain complex interactions in detail. "
        "Length is not a concern — completeness is.",
    ),
    (
        "analyst_historical_context",
        "Investigate the historical background, milestones, and evolution of a topic.",
        "You are a historical context analyst. Produce a COMPREHENSIVE "
        "historical timeline and analysis covering:\n"
        "1. ORIGINS: When and how did this field/topic begin? Who were the "
        "pioneers and what motivated them?\n"
        "2. KEY MILESTONES: Create a detailed chronological timeline of every "
        "major breakthrough, publication, policy change, or turning point. "
        "Include specific dates, names, and institutions.\n"
        "3. EVOLUTION OF THINKING: How has the scientific/expert consensus "
        "changed over time? What were the major paradigm shifts?\n"
        "4. FAILED APPROACHES: What was tried and didn't work? Why did it fail?\n"
        "5. INSTITUTIONAL HISTORY: Which organizations, companies, or governments "
        "have been most influential and how has their role changed?\n"
        "6. LESSONS LEARNED: What does the history tell us about likely future "
        "developments?\n\n"
        "Be comprehensive and specific. Include dates, names, and details "
        "throughout. This should read like a thorough review article.",
    ),
    (
        "analyst_expert_synthesis",
        "Provide a domain-expert-level perspective with deep technical insights.",
        "You are an expert synthesis analyst writing a LONG-FORM technical "
        "review. Produce a comprehensive, authoritative analysis that covers:\n"
        "1. TECHNICAL FOUNDATIONS: Explain the core science/engineering at "
        "a graduate-level depth. Use precise terminology.\n"
        "2. STATE OF THE ART: What are the current best results, leading "
        "groups, and benchmark performances? Be specific with numbers.\n"
        "3. OPEN PROBLEMS: What are the unsolved challenges? Why are they hard?\n"
        "4. COMPETING APPROACHES: How do different research groups/companies "
        "approach this differently? What are the trade-offs?\n"
        "5. INTERDISCIPLINARY CONNECTIONS: How does this connect to adjacent "
        "fields? What cross-pollination is happening?\n"
        "6. PRACTICAL IMPLICATIONS: What does this mean for real-world "
        "applications, policy, or industry?\n"
        "7. CRITICAL ASSESSMENT: Where is the hype outpacing reality? What "
        "claims should be treated with skepticism?\n\n"
        "Write at length. This should be the definitive expert briefing on "
        "the topic — thorough enough to bring a knowledgeable reader fully "
        "up to speed. Do not abbreviate or summarize prematurely.",
    ),
]


# ---------------------------------------------------------------------------
# Build schemas and dispatch table
# ---------------------------------------------------------------------------

def _make_analyst_fn(name: str, system_prompt: str):
    """Create an async tool function for one analyst."""
    async def _analyst(arguments: dict, agent_id: str,
                       group_id: str = None) -> str:
        resp = await llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arguments["question"]},
            ],
            agent_id=agent_id,
            call_type=name,
            detail=arguments["question"][:40],
            group_id=group_id,
        )
        return resp.choices[0].message.content
    return _analyst


ANALYST_SCHEMAS = []
ANALYST_TOOLS = {}

for _name, _desc, _sys in _ANALYSTS:
    ANALYST_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": _name,
            "description": _desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific research question to investigate",
                    },
                },
                "required": ["question"],
            },
        },
    })
    ANALYST_TOOLS[_name] = _make_analyst_fn(_name, _sys)
