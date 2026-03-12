"""Tool definitions and dispatch for the orchestrator agent.

The orchestrator LLM calls these tools via OpenAI function calling.
Each tool makes a single LLM call with a specialized system prompt,
routed through the scheduler via llm_call().

Tools fall into two categories:
  - Analysts (9): investigate a research question from a specific angle
  - Reviewers (3): critique a draft from a specific perspective
"""

from tools.analysts import ANALYST_SCHEMAS, ANALYST_TOOLS
from tools.reviewers import REVIEWER_SCHEMAS, REVIEWER_TOOLS

ALL_TOOL_SCHEMAS = ANALYST_SCHEMAS + REVIEWER_SCHEMAS

_DISPATCH = {**ANALYST_TOOLS, **REVIEWER_TOOLS}


async def execute_tool(name: str, arguments: dict, agent_id: str,
                       group_id: str = None) -> str:
    """Dispatch a tool call to the appropriate analyst or reviewer."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return await fn(arguments, agent_id, group_id=group_id)
