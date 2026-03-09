import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def send_heavy_request(req_id):
    start = time.monotonic()
    # Using 'with_raw_response' is the only way to get the headers reliably
    response = await client.chat.completions.with_raw_response.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "count to 100"}],
        max_tokens=4000  # High reservation
    )
    end = time.monotonic()
    
    headers = response.headers
    remaining_t = headers.get("x-ratelimit-remaining-tokens")
    remaining_r = headers.get("x-ratelimit-remaining-requests")
    
    print(f"Req {req_id} | Latency: {end-start:.2f}s | TPM Left: {remaining_t} | RPM Left: {remaining_r}")

async def main():
    # Fire 40 requests concurrently to overcome the ~33k/sec refill rate
    tasks = [send_heavy_request(i) for i in range(40)]
    await asyncio.gather(*tasks)

asyncio.run(main())