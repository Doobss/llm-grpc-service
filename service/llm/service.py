import logging

import grpc
from typing import AsyncIterator
from service.llm.grpc.service_pb2 import DESCRIPTOR
from service.llm.generation import ModelProcess, GenerationPrompt
from service.llm.grpc.prompt_pb2 import PromptRequest, PromptReply, PromptMetaData
from service.llm.grpc.service_pb2_grpc import LlmServicer, add_LlmServicer_to_server


class LlmService(LlmServicer):

    model: ModelProcess = None

    def __init__(self, model: ModelProcess, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    async def prompt(self, request: PromptRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[PromptReply]:
        logging.info(f'llm.prompt {request.id = } | {request.content = } | {request.config = }')
        message = GenerationPrompt(
            prompt=request.content,
            id=request.id
        )
        model_info = self.model.model_info
        total_chunks, total_batch_size, total_tokens, total_process_time = 0, 0, 0, 0
        async for chunk in self.model.prompt(message=message):
            total_chunks += 1
            total_batch_size += chunk.batch_size
            total_tokens += self.model.config.chunk_size
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
    add_LlmServicer_to_server(LlmService(
        model=ModelProcess()
    ), server)
    return server


def llm_service_descriptor() -> str:
    return DESCRIPTOR.services_by_name['Llm'].full_name
