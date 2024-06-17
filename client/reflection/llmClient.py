import grpc
import logging
from uuid import uuid4
from reflection.grpcReflection import (
    Service,
    GrpcReflection
)
from typing import Iterator, Type
from google.protobuf.message import Message
from reflection.serviceReflection import ServiceReflection, ReflectedMethod

logger = logging.getLogger(__name__)



class LLmClient(ServiceReflection):

    name: str = 'Llm'
    reflection: GrpcReflection
    llm_service: Service
    _prompt_method: ReflectedMethod

    def __init__(self, reflection: GrpcReflection) -> None:
        super().__init__(reflection=reflection)
        self._prompt_method = self._get_method('prompt')

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
