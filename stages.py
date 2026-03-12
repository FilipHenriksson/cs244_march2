"""Workflow stage definitions for the research orchestration pipeline.

Each WorkflowStage subclass encapsulates the call_type, id_suffix, and
message_factory() for one LLM call in the pipeline.  Stage 2 (the agent
fan-out) is excluded — it involves a multi-turn tool-use loop handled
separately in agent.py.
"""

from abc import ABC, abstractmethod

NUM_AGENTS = 5


class WorkflowStage(ABC):
    """Base class for a single-call pipeline stage."""

    call_type: str
    id_suffix: str

    @abstractmethod
    def message_factory(self, **kwargs) -> list[dict]:
        """Build the chat-completion messages list for this stage."""
        ...




# ---------------------------------------------------------------------------
# Stage 1 — Breakdown
# ---------------------------------------------------------------------------

class BreakdownStage(WorkflowStage):
    call_type = "breakdown"
    id_suffix = "orch"

    def message_factory(self, *, prompt: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    f"You are a research coordinator. Given a research topic, generate exactly "
                    f"{NUM_AGENTS} distinct sub-questions that together cover the topic comprehensively. "
                    f"Return ONLY the questions, one per line, numbered 1-{NUM_AGENTS}."
                ),
            },
            {"role": "user", "content": prompt},
        ]


# ---------------------------------------------------------------------------
# Stage 3 — Synthesis (fan-in after agent results)
# ---------------------------------------------------------------------------

class SynthesisStage(WorkflowStage):
    call_type = "synthesis"
    id_suffix = "orch"

    def message_factory(self, *, prompt: str, findings: str) -> list[dict]:
        return [
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
        ]


# ---------------------------------------------------------------------------
# Stage 4 — Reviewer (one instance per review pass)
# ---------------------------------------------------------------------------

class ReviewerStage(WorkflowStage):
    def __init__(self, *, call_type: str, id_suffix: str, system_prompt: str):
        self.call_type = call_type
        self.id_suffix = id_suffix
        self._system_prompt = system_prompt

    def message_factory(self, *, prompt: str, synthesis_text: str) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"Original research topic: {prompt}\n\n"
                    f"Synthesis to review:\n{synthesis_text}"
                ),
            },
        ]


REVIEWERS = [
    ReviewerStage(
        call_type="review_citations",
        id_suffix="r0",
        system_prompt=(
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
    ),
    ReviewerStage(
        call_type="review_style",
        id_suffix="r1",
        system_prompt=(
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
    ),
    ReviewerStage(
        call_type="review_facts",
        id_suffix="r2",
        system_prompt=(
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
    ),
]


# ---------------------------------------------------------------------------
# Stage 5 — Finalize (fan-in after reviewer feedback)
# ---------------------------------------------------------------------------

class FinalizeStage(WorkflowStage):
    call_type = "finalize"
    id_suffix = "orch"

    def message_factory(self, *, prompt: str, synthesis_text: str,
                        review_feedback: str) -> list[dict]:
        return [
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
        ]
