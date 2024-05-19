import logging

import grpc
from typing import AsyncIterator
from ...llm import ModelProcess, GenerationPrompt
from ..llm.llm_pb2 import PromptRequest, PromptReply, PromptMetaData
from ..llm.llm_pb2_grpc import LlmServicer, add_LlmServicer_to_server


class LlmService(LlmServicer):

    model: ModelProcess = None

    async def prompt(self, request: PromptRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[PromptReply]:
        logging.info(f'llm.prompt {request.id = } | {request.content = } | {request.config = }')
        model = self.model or ModelProcess()
        self.model = model
        message = GenerationPrompt(
            prompt=request.content,
            id=request.id
        )
        model_info = model.model_info
        total_chunks, total_batch_size, total_tokens, total_process_time = 0, 0, 0, 0
        async for chunk in self.model.prompt(message=message):
            total_chunks += 1
            total_batch_size += chunk.batch_size
            total_tokens += model.config.chunk_size
            total_process_time += chunk.process_time
            yield PromptReply(
                id=chunk.id,
                content=chunk.generated,
                config=request.config,
                meta=PromptMetaData(
                    tokens_per_second=(total_tokens / total_process_time),
                    average_batch_size=(total_batch_size / total_chunks),
                    model=PromptMetaData.Model(
                        device=model_info['device'],
                        dtype=model_info['dtype'],
                        type=model_info['type'],
                        number_of_devices=model_info['number_of_devices']
                    )
                ),
                is_end_of_sequence=chunk.reached_eos
            )


def add_llm_service(server: grpc.aio.Server) -> grpc.aio.Server:
    add_LlmServicer_to_server(LlmService(), server)
    return server
