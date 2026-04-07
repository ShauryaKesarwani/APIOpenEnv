import asyncio
from client import ApiOpenEnv
from models import ApiOpenAction

async def main():
    async with ApiOpenEnv(base_url="http://localhost:8000") as env:
        r = await env.reset()
    print("reset:", r.observation.echoed_message)

    for msg in ["hello", "rl step", "final"]:
        r = await env.step(ApiOpenAction(message=msg))
        print(r.observation.echoed_message, r.reward, r.done)

if __name__ == "__main__":
    asyncio.run(main())