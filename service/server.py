import asyncio
import logging

import grpc
from grpc_reflection.v1alpha import reflection
from service.llm import add_llm_service, llm_service_descriptor


def _setup_reflection(server: grpc.aio.Server) -> None:
    logging.info(f'Setting up reflection')
    service_names: tuple[str, str] = (
        llm_service_descriptor(),
        reflection.SERVICE_NAME
    )
    reflection.enable_server_reflection(service_names, server)


async def serve() -> None:
    server = grpc.aio.server()
    server = add_llm_service(server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)

    _setup_reflection(server)

    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


def start() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    start()
