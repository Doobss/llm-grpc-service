import asyncio
import logging

from service.grpc import ClientBase
from typing import AsyncIterator
from service.llm.grpc.llm_pb2 import PromptRequest, PromptConfig, PromptReply
from service.llm.grpc.llm_pb2_grpc import LlmStub


class LlmClient(ClientBase):

    stub: LlmStub = None

    async def prompt(self, request: PromptRequest) -> AsyncIterator[PromptReply]:
        async with self.channel() as channel:
            stub = LlmStub(channel=channel)
            stream: AsyncIterator[PromptReply] = stub.prompt(request)
            async for response in stream:
                yield response
                if response.is_end_of_sequence:
                    break


async def run() -> None:
    client = LlmClient()
    request = PromptRequest(
            id="test_id",
            content="Hello",
            config=PromptConfig()
        )
    # Read from an async generator
    async for response in client.prompt(request=request):
        print(f"Greeter client received from async generator: \n{response = }")

if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(run())
