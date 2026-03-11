"""Research topics used as workload inputs for single and batch runs."""

RESEARCH_PROMPTS = [
    "What are the latest advances in quantum error correction?",
    "How does CRISPR-Cas9 compare to base editing for therapeutic use?",
    "What is the current state of nuclear fusion energy research?",
    "How are large language models being used in drug discovery?",
    "What are the environmental impacts of deep-sea mining?",
]

DEFAULT_PROMPT = RESEARCH_PROMPTS[0]


# Reviewer configurations — each defines a parallel review pass over the
# synthesized draft.  id_suffix builds the agent_id for tracing; 
# call_type used for trace log.

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