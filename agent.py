"""Research agent — tool-use loop over a single sub-question.

Each agent runs up to MAX_TOOL_ROUNDS iterations:
  1. Send conversation history to the LLM with tool schemas
  2. If the LLM returns tool calls -> execute them, append results, repeat
  3. If the LLM returns plain text -> return it as the agent's findings

Tools (web_search, summarize, analyze) are defined in tools.py; each makes
its own LLM call, so a single agent round may produce multiple scheduler
submissions.
"""

from llm import llm_call
from tools import TOOL_SCHEMAS, execute_tool

MAX_TOOL_ROUNDS = 3


async def run_agent(sub_question: str, agent_id: int, session_id: int = 0,
                    group_id: str = None) -> str:
    """Run a single research agent that can use tools in a loop."""
    aid = f"s{session_id}:a{agent_id}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Use the available tools to research "
                "the given question. Search for information, summarize findings, and "
                "analyze results. After gathering enough information (1-3 tool calls), "
                "provide your final findings as a text response."
            ),
        },
        {"role": "user", "content": sub_question},
    ]

    for round_num in range(MAX_TOOL_ROUNDS):
        resp = await llm_call(
            messages=messages,
            agent_id=aid,
            call_type="agent_turn",
            detail=f"round-{round_num}",
            group_id=group_id,
            tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            result = await execute_tool(
                tool_call.function.name,
                tool_call.function.arguments,
                aid,
                group_id=group_id,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Final response after max rounds
    resp = await llm_call(
        messages=messages,
        agent_id=aid,
        call_type="agent_turn",
        detail="final",
        group_id=group_id,
    )
    return resp.choices[0].message.content
