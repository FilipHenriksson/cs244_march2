"""Example client — drives the strict-mode agent loop over the scheduling proxy.

Demonstrates the full research pipeline (11 LLM calls) using only HTTP
requests to the proxy server.  The client manages conversation state and
fan-out/fan-in barriers; the server handles scheduling and rate limiting.

Usage::

    # Terminal 1: start the proxy
    hypercorn api:app --host 0.0.0.0 --port 8000

    # Terminal 2: run this client
    python client_example.py
    python client_example.py --prompt "nuclear fusion energy" --base-url http://localhost:8000
    python client_example.py --sessions 5 --stagger 4.0
"""

import argparse
import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

from prompts.system import ORCHESTRATOR_SYSTEM_PROMPT
from tools.analysts import _ANALYSTS
from tools.reviewers import _REVIEWERS

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("PROXY_API_KEY", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

STRICT_ANALYSTS = [
    "analyst_web_research", "analyst_summarizer", "analyst_deep_analysis",
    "analyst_historical_context", "analyst_statistical",
]
STRICT_REVIEWERS = ["review_citations", "review_style", "review_facts"]

ALL_CALL_TYPES: dict[str, str] = {
    "orchestrator": ORCHESTRATOR_SYSTEM_PROMPT,
}
for name, _desc, system_prompt in _ANALYSTS:
    ALL_CALL_TYPES[name] = system_prompt
for name, _desc, system_prompt in _REVIEWERS:
    ALL_CALL_TYPES[name] = system_prompt


async def register_call_types(client: httpx.AsyncClient):
    """Register all call types with the proxy (idempotent)."""
    for name, system_prompt in ALL_CALL_TYPES.items():
        resp = await client.post("/call_types", json={
            "name": name,
            "system_prompt": system_prompt,
        })
        resp.raise_for_status()
    print(f"Registered {len(ALL_CALL_TYPES)} call types")


async def run_session(client: httpx.AsyncClient, prompt: str,
                      session_id: str | None = None) -> dict:
    """Run one strict-mode research session over the proxy.

    Returns a dict with the final synthesis text, timing, and call counts.
    """
    start = time.time()

    if session_id is None:
        resp = await client.post("/sessions")
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

    base = f"/sessions/{session_id}/completions"
    messages: list[dict] = []

    # Step 1: Orchestrator plans the research
    messages.append({"role": "user", "content": prompt})
    resp = await client.post(base, json={
        "call_type": "orchestrator",
        "call_detail": "plan",
        "messages": list(messages),
    })
    resp.raise_for_status()
    plan = resp.json()["content"]
    messages.append({"role": "assistant", "content": plan})
    print(f"  Step 1 (plan): {len(plan)} chars")

    # Step 2: Fan out 5 analysts in parallel via batch endpoint
    analyst_calls = [
        {
            "call_type": name,
            "messages": [{"role": "user", "content": prompt}],
        }
        for name in STRICT_ANALYSTS
    ]
    resp = await client.post(f"{base}/batch", json={"calls": analyst_calls})
    resp.raise_for_status()
    analyst_results = resp.json()["completions"]
    analyst_text = "\n\n".join(
        f"--- {name} ---\n{r['content']}"
        for name, r in zip(STRICT_ANALYSTS, analyst_results)
    )
    messages.append({"role": "user", "content":
        f"Here are findings from 5 specialist analysts:\n\n{analyst_text}\n\n"
        f"Write a comprehensive draft synthesis of these findings."
    })
    print(f"  Step 2 (analysts x{len(STRICT_ANALYSTS)}): "
          f"{sum(len(r['content']) for r in analyst_results)} chars total")

    # Step 3: Orchestrator drafts synthesis
    resp = await client.post(base, json={
        "call_type": "orchestrator",
        "call_detail": "synthesize",
        "messages": list(messages),
    })
    resp.raise_for_status()
    draft = resp.json()["content"]
    messages.append({"role": "assistant", "content": draft})
    print(f"  Step 3 (synthesis): {len(draft)} chars")

    # Step 4: Fan out 3 reviewers in parallel via batch endpoint
    reviewer_calls = [
        {
            "call_type": name,
            "messages": [{"role": "user", "content": draft}],
        }
        for name in STRICT_REVIEWERS
    ]
    resp = await client.post(f"{base}/batch", json={"calls": reviewer_calls})
    resp.raise_for_status()
    reviewer_results = resp.json()["completions"]
    reviewer_text = "\n\n".join(
        f"--- {name} ---\n{r['content']}"
        for name, r in zip(STRICT_REVIEWERS, reviewer_results)
    )
    messages.append({"role": "user", "content":
        f"Here is feedback from 3 reviewers:\n\n{reviewer_text}\n\n"
        f"Incorporate the feedback and produce your final polished synthesis."
    })
    print(f"  Step 4 (reviewers x{len(STRICT_REVIEWERS)}): "
          f"{sum(len(r['content']) for r in reviewer_results)} chars total")

    # Step 5: Orchestrator final synthesis
    resp = await client.post(base, json={
        "call_type": "orchestrator",
        "call_detail": "final",
        "messages": list(messages),
    })
    resp.raise_for_status()
    final = resp.json()["content"]
    elapsed = time.time() - start
    print(f"  Step 5 (final): {len(final)} chars — total {elapsed:.1f}s")

    await client.delete(f"/sessions/{session_id}")

    n_analysts = len(STRICT_ANALYSTS)
    n_reviewers = len(STRICT_REVIEWERS)
    return {
        "text": final,
        "elapsed": elapsed,
        "llm_calls": 3 + n_analysts + n_reviewers,
        "tool_calls": n_analysts + n_reviewers,
    }


async def main():
    parser = argparse.ArgumentParser(description="Client for the LLM scheduling proxy")
    parser.add_argument("--prompt", type=str,
                        default="What are the latest advances in quantum error correction?")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--sessions", type=int, default=1,
                        help="Number of concurrent sessions to launch")
    parser.add_argument("--stagger", type=float, default=0.0,
                        help="Seconds between session launches (0 = all at once)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="HTTP request timeout in seconds")
    args = parser.parse_args()

    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=httpx.Timeout(args.timeout),
        http2=True,
        headers=AUTH_HEADERS,
    ) as client:
        await register_call_types(client)

        if args.sessions == 1:
            print(f"\nRunning single session: {args.prompt[:60]}...")
            result = await run_session(client, args.prompt)
            print(f"\nDone: {result['llm_calls']} LLM calls, "
                  f"{result['elapsed']:.1f}s")
            print(f"\n{'='*60}")
            print(result["text"][:500])
            if len(result["text"]) > 500:
                print(f"... ({len(result['text'])} chars total)")
        else:
            print(f"\nLaunching {args.sessions} sessions "
                  f"(stagger={args.stagger}s)...\n")

            async def _launch(i: int):
                if args.stagger > 0:
                    await asyncio.sleep(i * args.stagger)
                print(f"[session {i}] starting")
                result = await run_session(client, args.prompt)
                print(f"[session {i}] done in {result['elapsed']:.1f}s")
                return result

            results = await asyncio.gather(*[_launch(i) for i in range(args.sessions)])
            durations = [r["elapsed"] for r in results]
            print(f"\nAll {args.sessions} sessions complete")
            print(f"  Mean: {sum(durations)/len(durations):.1f}s")
            print(f"  Min:  {min(durations):.1f}s")
            print(f"  Max:  {max(durations):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
