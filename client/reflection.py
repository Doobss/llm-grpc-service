import os
import grpc
from typing import Type, Union
from google.protobuf.message import Message
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import MessageFactory
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import ProtoReflectionDescriptorDatabase
from google.protobuf.descriptor import (
    ServiceDescriptor,
    MethodDescriptor,
    Descriptor
)


class Described:

    descriptor: Descriptor

    @property
    def name(self) -> str:
        return self.descriptor.name


class Method(Described):

    descriptor: MethodDescriptor
    messages: dict[str, Type[Message]]

    def __init__(self, descriptor: MethodDescriptor, descriptor_pool: DescriptorPool) -> None:
        self.descriptor = descriptor
        factory = MessageFactory(descriptor_pool)
        self.messages = {
            "input": factory.GetPrototype(
                descriptor=descriptor_pool.FindMessageTypeByName(
                    descriptor.input_type.full_name
                )
            ) if isinstance(descriptor.input_type, Descriptor) else None,
            "output": factory.GetPrototype(
                descriptor=descriptor_pool.FindMessageTypeByName(
                    descriptor.output_type.full_name
                )
            ) if isinstance(descriptor.output_type, Descriptor) else None,
        }
        for name, message in self.messages.items():
            setattr(self, name, message)

    @property
    def path(self) -> str:
        return f'/{self.descriptor.containing_service.full_name}/{self.name}'

    def __repr__(self) -> str:
        return f'Method(' \
               f'\n    name: {self.name}' \
               f'\n    input: {self.messages["input"].__qualname__ if self.messages["input"] is not None else None}' \
               f'\n    output: {self.messages["output"].__qualname__ if self.messages["output"] is not None else None}' \
               f'\n)'


class Service(Described):

    descriptor: ServiceDescriptor
    methods: dict[str, Method]

    def __init__(self, descriptor: ServiceDescriptor, descriptor_pool: DescriptorPool) -> None:
        self.descriptor = descriptor
        self.methods = {
            name: Method(
                descriptor=method_descriptor,
                descriptor_pool=descriptor_pool
            ) for name, method_descriptor in descriptor.methods_by_name.items()
        }
        for method in self.methods.values():
            setattr(self, method.name, method)

    def __repr__(self) -> str:
        methods = ",".join([f"\n\t\t{method}" for method in
                            ["\n\t\t".join(str(method).split("\n")) for method in self.methods.values()]])
        return f'Service(' \
               f'\n    name: {self.name}' \
               f'\n    methods: [{methods}' \
               f'\n    ]' \
               f'\n)'


class GrpcReflection:

    target: str
    services: dict[str, Service]
    descriptor_pool: DescriptorPool
    channel: grpc.Channel
    reflection_db: ProtoReflectionDescriptorDatabase
    credentials: Union[grpc.ChannelCredentials, None] = None

    def __init__(self, target: str = None, credentials: grpc.ChannelCredentials = None) -> None:
        self.target = target or os.getenv("GRPC_TARGET", default="127.0.0.1:50051")
        self.credentials = credentials
        self.channel = _channel(
            target=self.target,
            credentials=self.credentials
        )
        self.reflection_db = ProtoReflectionDescriptorDatabase(
            channel=self.channel
        )
        self.descriptor_pool = DescriptorPool(descriptor_db=self.reflection_db)
        self.services: dict[str, Service] = {
            name: Service(
                descriptor=self.descriptor_pool.FindServiceByName(name),
                descriptor_pool=self.descriptor_pool
            )
            for name in self.reflection_db.get_services()
        }
        for service in self.services.values():
            setattr(self, service.name, service)

    def __repr__(self) -> str:

        services = ",".join([f"\n\t\t{service}" for service in [
            "\n\t\t".join(str(service).split('\n')) for service in self.services.values()]])
        return f'GrpcReflection(' \
               f'\n    services: [{services}' \
               f'\n    ]' \
               f'\n)'


def _channel(
        target: str = "127.0.0.1:50051",
        credentials: grpc.ChannelCredentials = None
) -> grpc.Channel:
    return grpc.insecure_channel(
        target=target
    ) if credentials is None else grpc.secure_channel(
        target=target,
        credentials=credentials
    )


def _reflection_db(
        target: str = "127.0.0.1:50051",
        credentials: grpc.ChannelCredentials = None
) -> ProtoReflectionDescriptorDatabase:
    return ProtoReflectionDescriptorDatabase(
        channel=_channel(
            target=target,
            credentials=credentials
        )
    )


def _descriptor_pool(
        target: str = "127.0.0.1:50051",
        credentials: grpc.ChannelCredentials = None
) -> DescriptorPool:
    return DescriptorPool(
        descriptor_db=_reflection_db(
            target=target,
            credentials=credentials
        )
    )


def main() -> None:
    grpc_reflection = GrpcReflection(
        target=None,
        credentials=None
    )

    print(grpc_reflection)

    llm_service: Service = grpc_reflection.Llm if hasattr(grpc_reflection, 'Llm') else None
    if llm_service is None:
        raise ValueError(f"Missing Llm service")

    prompt_method: Method = llm_service.prompt if hasattr(llm_service, 'prompt') else None
    if prompt_method is None:
        raise ValueError(f"Missing prompt method on Llm service")

    prompt_request = prompt_method.input if hasattr(prompt_method, "input") else None
    prompt_response = prompt_method.output if hasattr(prompt_method, "output") else None
    if prompt_request is None:
        raise ValueError(f"Missing prompt request for prompt method on Llm service")

    if prompt_response is None:
        raise ValueError(f"Missing prompt response for prompt method on Llm service")

    prompt = grpc_reflection.channel.unary_stream(
        method=prompt_method.path,
        request_serializer=prompt_request.SerializeToString,
        response_deserializer=prompt_response.FromString
    )
    for message in prompt(prompt_request(
        id="test",
        content="GPT4 Correct User:Your name is Genome <|end_of_turn|> GPT4 Correct Assistant: Yes, I understand my job.<|end_of_turn|> GPT4 Correct User: Great! query: What are the best cheeses? GPT4 Correct Assistant: response:"
    )):
        print(f'{message = }')

    print()


if __name__ == "__main__":
    main()
