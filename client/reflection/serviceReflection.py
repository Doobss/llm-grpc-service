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

logger = logging.getLogger()


@dataclass
class ReflectedMethod:
    method: Method
    request: Type[Message]
    response: Type[Message]


class ServiceReflection:

    name: str
    reflection: GrpcReflection
    service: Service
    _apply_template_method: ReflectedMethod
    _get_template_method: ReflectedMethod


    def __init__(self, reflection: GrpcReflection) -> None:
        self.reflection = reflection
        if self.name is None or self.name == '':
            raise ValueError(f'Missing service name on class definition.')
        service_name = self.name
        logger.info(f'Checking {service_name} reflection for required values')
        self.service: Service = getattr(self.reflection, service_name) if hasattr(self.reflection, service_name) else None


    def _get_method(self, method_name: str) -> ReflectedMethod:
        method: Method = getattr(self.service, method_name) if hasattr(self.service, method_name) else None
        if method is None:
            raise ValueError(f"Missing {method_name} method on {self.name} service")
        request = method.input if hasattr(method, "input") else None
        response = method.output if hasattr(method, "output") else None
        if request is None:
            raise ValueError(f"Missing request for {method_name} on {self.name} service")
        if response is None:
            raise ValueError(f"Missing response for {method_name} on {self.name} service")
        return ReflectedMethod(
            method=method,
            request=request,
            response=response
        )


if __name__ == '__main__':
    ...
