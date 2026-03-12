import asyncio
import time
from openai import AsyncOpenAI
from llm import RATELIMIT_HEADERS
import random

client = AsyncOpenAI()
async def send_heavy_request(req_id):
    start = time.monotonic()
    # Using 'with_raw_response' is the only way to get the headers reliably
    count_val = random.randint(10000, 100000)
    response = await client.chat.completions.with_raw_response.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": f"count to {count_val}"}],
        max_tokens=4096  # High reservation
    )
    end = time.monotonic()
    
    headers = response.headers

    # 2. Parse it to get the standard Completion object
    completion = response.parse()

    # Access the text as usual
    content = completion.choices[0].message.content
    print(content)

    for header in RATELIMIT_HEADERS:
        print(f"Req {req_id} | {header}: {headers.get(header)}")

async def main():
    # Fire 40 requests concurrently to overcome the ~33k/sec refill rate
    tasks = [send_heavy_request(i) for i in range(2)]
    await asyncio.gather(*tasks)

asyncio.run(main())