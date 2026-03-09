import asyncio
from dotenv import load_dotenv
load_dotenv()

from llm import fetch_ratelimit_headers


async def main():
    print("Sending minimal request to OpenAI...")
    headers = await fetch_ratelimit_headers()

    print("\nRate limit headers:")
    for key, value in headers.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
