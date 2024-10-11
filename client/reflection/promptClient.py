import grpc
import logging
from uuid import uuid4
from reflection.grpcReflection import (
    Method,
    Service,
    GrpcReflection
)
from reflection.serviceReflection import ServiceReflection, ReflectedMethod
from dataclasses import dataclass
from typing import Iterator, Type
from google.protobuf.message import Message

logger = logging.getLogger(__name__)


class PromptClient(ServiceReflection):

    name: str = 'Prompt'
    reflection: GrpcReflection
    service: Service
    _apply_template_method: ReflectedMethod
    _get_template_method: ReflectedMethod


    def __init__(self, reflection: GrpcReflection) -> None:
        super().__init__(reflection=reflection)

        self._apply_template_method = self._get_method('apply_template')
        self._get_template_method = self._get_method('get_template')


    def apply_template(self, messages: list[dict]) -> Message:
        method = self._apply_template_method.method
        request = self._apply_template_method.request
        response = self._apply_template_method.response

        method = self.reflection.channel.unary_unary(
            method=method.path,
            request_serializer=request.SerializeToString,
            response_deserializer=response.FromString
        )
        custom_template = '''{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ ' assistant:\n' }}{% endif %}{% endfor %}'''
        return method(request(
            messages=messages,
            custom_template=custom_template
        ))

    def get_template(self) -> Message:
        method = self._get_template_method.method
        request = self._get_template_method.request
        response = self._get_template_method.response

        method = self.reflection.channel.unary_unary(
            method=method.path,
            request_serializer=request.SerializeToString,
            response_deserializer=response.FromString
        )
        return method(request())


if __name__ == '__main__':
    client = PromptClient()
    content = 'GPT4 Correct User:Your name is Genome <|end_of_turn|> ' \
              'GPT4 Correct Assistant: Yes, I understand my job.<|end_of_turn|> ' \
              'GPT4 Correct User: Great! query: What are the best cheeses? ' \
              'GPT4 Correct Assistant: response:'
    for message in client.prompt(content=content):
        logger.info(f'{message = }')
