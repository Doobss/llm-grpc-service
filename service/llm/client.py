from service.grpc import ClientBase
from typing import AsyncIterator
from service.llm.grpc.prompt_pb2 import PromptRequest, PromptConfig, PromptReply
from service.llm.grpc.service_pb2_grpc import LlmStub


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
