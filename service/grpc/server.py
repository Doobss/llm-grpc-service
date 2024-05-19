import asyncio
import logging

import grpc
from service.grpc.llm.llmService import add_llm_service


async def serve() -> None:
    server = grpc.aio.server()
    server = add_llm_service(server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


def start() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    start()
