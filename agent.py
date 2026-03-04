from llm import llm_call
from tools import TOOL_SCHEMAS, execute_tool

MAX_TOOL_ROUNDS = 3


async def run_agent(sub_question: str, agent_id: int, session_id: int = 0) -> str:
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
    )
    return resp.choices[0].message.content
