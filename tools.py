import json
from llm import llm_call

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic. Returns relevant search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize a piece of text into key points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to summarize"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": "Analyze text in the context of a specific question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to analyze"},
                    "question": {"type": "string", "description": "The question to answer about the text"},
                },
                "required": ["text", "question"],
            },
        },
    },
]


async def web_search(query: str, agent_id: str) -> str:
    resp = await llm_call(
        messages=[
            {"role": "system", "content": "You simulate a web search engine. Given a query, return 3-5 realistic search result snippets with titles and brief excerpts. Be informative and factual."},
            {"role": "user", "content": query},
        ],
        agent_id=agent_id,
        call_type="web_search",
        detail=query[:40],
    )
    return resp.choices[0].message.content


async def summarize(text: str, agent_id: str) -> str:
    resp = await llm_call(
        messages=[
            {"role": "system", "content": "Summarize the following text into concise key points."},
            {"role": "user", "content": text},
        ],
        agent_id=agent_id,
        call_type="summarize",
        detail=f"len={len(text)}",
    )
    return resp.choices[0].message.content


async def analyze(text: str, question: str, agent_id: str) -> str:
    resp = await llm_call(
        messages=[
            {"role": "system", "content": "Analyze the provided text to answer the given question. Be thorough and specific."},
            {"role": "user", "content": f"Text:\n{text}\n\nQuestion: {question}"},
        ],
        agent_id=agent_id,
        call_type="analyze",
        detail=question[:40],
    )
    return resp.choices[0].message.content


async def execute_tool(name: str, arguments: str, agent_id: str) -> str:
    args = json.loads(arguments)
    if name == "web_search":
        return await web_search(args["query"], agent_id)
    elif name == "summarize":
        return await summarize(args["text"], agent_id)
    elif name == "analyze":
        return await analyze(args["text"], args["question"], agent_id)
    else:
        return f"Unknown tool: {name}"
