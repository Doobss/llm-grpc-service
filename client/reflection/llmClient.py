import grpc
import logging
from uuid import uuid4
from reflection.grpcReflection import (
    Method,
    Service,
    GrpcReflection
)
from dataclasses import dataclass
from typing import Iterator, Type
from google.protobuf.message import Message

logger = logging.getLogger(__name__)


@dataclass
class _PromptMethod:
    method: Method
    request: Type[Message]
    response: Type[Message]


class LLmClient:

    reflection: GrpcReflection
    llm_service: Service
    _prompt_method: _PromptMethod

    def __init__(self, target: str = None, credentials: grpc.ChannelCredentials = None) -> None:
        logging.basicConfig(level=logging.INFO)
        logger.info(f'Starting Llm client using grpc reflection. {target = } | {bool(credentials) = }')
        self.reflection = GrpcReflection(
            target=target,
            credentials=credentials
        )
        logger.info(f'{self.reflection}')
        logger.info('Checking reflection for required values')
        self.llm_service: Service = self.reflection.Llm if hasattr(self.reflection, 'Llm') else None
        if self.llm_service is None:
            raise ValueError(f"Missing Llm service")

        prompt_method: Method = self.llm_service.prompt if hasattr(self.llm_service, 'prompt') else None
        if prompt_method is None:
            raise ValueError(f"Missing prompt method on Llm service")

        prompt_request = prompt_method.input if hasattr(prompt_method, "input") else None
        prompt_response = prompt_method.output if hasattr(prompt_method, "output") else None
        if prompt_request is None:
            raise ValueError(f"Missing prompt request for prompt method on Llm service")

        if prompt_response is None:
            raise ValueError(f"Missing prompt response for prompt method on Llm service")
        self._prompt_method = _PromptMethod(
            method=prompt_method,
            request=prompt_request,
            response=prompt_response
        )

    def prompt(self, content: str, prompt_id: str = None) -> Iterator[Message]:
        prompt_method = self._prompt_method.method
        prompt_request = self._prompt_method.request
        prompt_response = self._prompt_method.response

        prompt = self.reflection.channel.unary_stream(
            method=prompt_method.path,
            request_serializer=prompt_request.SerializeToString,
            response_deserializer=prompt_response.FromString
        )
        request = prompt_request(
            id=prompt_id or str(uuid4()),
            content=content
        )
        for response in prompt(request):
            yield response
            if hasattr(response, 'is_end_of_sequence') and response.is_end_of_sequence:
                break


if __name__ == '__main__':
    client = LLmClient()
    content = 'GPT4 Correct User:Your name is Genome <|end_of_turn|> ' \
              'GPT4 Correct Assistant: Yes, I understand my job.<|end_of_turn|> ' \
              'GPT4 Correct User: Great! query: What are the best cheeses? ' \
              'GPT4 Correct Assistant: response:'
    for message in client.prompt(content=content):
        logger.info(f'{message = }')
