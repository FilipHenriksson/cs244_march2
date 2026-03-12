'''
This script sends a minimal chat completion request to OpenAI and prints the rate limit headers.
'''

import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
load_dotenv()


RATELIMIT_HEADERS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]

MODEL = "gpt-4.1-nano"
client = AsyncOpenAI()

async def fetch_ratelimit_headers(max_tokens: int = 1) -> dict[str, str]:
    """Send a minimal chat completion request and return OpenAI rate limit headers."""
    raw = await client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
    )
    return {h: raw.headers[h] for h in RATELIMIT_HEADERS if h in raw.headers}


async def main():
    print("Sending minimal request to OpenAI...")
    headers = await fetch_ratelimit_headers()

    print("\nRate limit headers:")
    for key, value in headers.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
