"""Three reviewer tools — each critiques a text from a specialized perspective.

Every reviewer is a single LLM call with a focused system prompt.  The prompts
are deliberately varied in scope: the style reviewer is fast (surface-level
checks), the citation reviewer is moderate, and the fact-checker is thorough
(must evaluate every claim).  This creates natural output-length variation.
"""

from llm import llm_call


# ---------------------------------------------------------------------------
# Reviewer definitions: (name, description, system_prompt)
#
# Ordered roughly from SHORT expected output to LONG expected output.
# ---------------------------------------------------------------------------

_REVIEWERS = [
    (
        "review_style",
        "Check grammar, tone, clarity, and structural quality.",
        "You are a style and grammar editor. Do a QUICK pass on the text. "
        "Only flag significant issues — ignore minor stylistic preferences. "
        "Format:\n"
        "- ISSUE: <description>\n"
        "  FIX: <specific rewrite>\n\n"
        "End with a one-sentence overall assessment. Keep your review concise "
        "— if the writing is clean, say so briefly and move on.",
    ),
    (
        "review_citations",
        "Check citation quality — flag unsourced or vaguely attributed claims.",
        "You are a citation and source checker. Your job is to:\n"
        "1. Identify every factual claim in the text.\n"
        "2. Check whether each claim is attributed to a source or left unsourced.\n"
        "3. Flag any claims that lack citations or have vague attribution "
        "(e.g., 'studies show' without specifics).\n"
        "4. Suggest where additional citations would strengthen the text.\n\n"
        "Return your findings as a structured list with the format:\n"
        "- CLAIM: <the claim>\n"
        "  STATUS: [cited | unsourced | vague]\n"
        "  SUGGESTION: <optional improvement>\n\n"
        "Be thorough — check EVERY factual claim in the document. "
        "End with a brief overall assessment of citation quality.",
    ),
    (
        "review_facts",
        "Fact-check claims — assess plausibility, flag incorrect or unverifiable statements.",
        "You are a rigorous fact checker. You must evaluate EVERY factual claim "
        "in the text — statistics, dates, names, causal claims, numerical "
        "comparisons, and attributions. For EACH claim:\n"
        "- CLAIM: <quote the exact claim>\n"
        "  VERDICT: [plausible | questionable | likely incorrect | unverifiable]\n"
        "  REASONING: <detailed explanation of why you assigned this verdict, "
        "including what the correct information is if the claim is wrong, or "
        "what specific aspects make it questionable>\n"
        "  CONFIDENCE: [high | medium | low]\n\n"
        "Do NOT skip any claims. Even if a claim seems obviously true, include "
        "it with a 'plausible' verdict. For questionable or incorrect claims, "
        "provide the correct information or explain what would need to be "
        "verified.\n\n"
        "End with:\n"
        "- SUMMARY: Overall factual accuracy score (e.g., '85% of claims are "
        "plausible') and the most critical issues found.\n"
        "- RECOMMENDATIONS: Specific claims that MUST be corrected or verified "
        "before publication.",
    ),
]


# ---------------------------------------------------------------------------
# Build schemas and dispatch table
# ---------------------------------------------------------------------------

def _make_reviewer_fn(name: str, system_prompt: str):
    """Create an async tool function for one reviewer."""
    async def _reviewer(arguments: dict, session_id: int) -> str:
        resp = await llm_call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": arguments["text"]},
            ],
            session_id=session_id,
            call_key=name,
            label=name,
        )
        return resp.choices[0].message.content
    return _reviewer


REVIEWER_SCHEMAS = []
REVIEWER_TOOLS = {}

for _name, _desc, _sys in _REVIEWERS:
    REVIEWER_SCHEMAS.append({
        "type": "function",
        "function": {
            "name": _name,
            "description": _desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to review",
                    },
                },
                "required": ["text"],
            },
        },
    })
    REVIEWER_TOOLS[_name] = _make_reviewer_fn(_name, _sys)
