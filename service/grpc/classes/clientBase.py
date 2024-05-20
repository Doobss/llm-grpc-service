import os
import grpc
from typing import Union


class ClientBase:

    user: str = None
    host: str = 'localhost'
    port: int = 50051
    secure: bool = False
    _channel: Union[grpc.aio.Channel, None] = None

    def __init__(self, host: str = None, port: int = None, user: str = None, secure: bool = None) -> None:
        self.host = host or os.getenv('GRPC_HOST', default='localhost')
        self.port = port or int(os.getenv('GRPC_PORT', default='50051'))
        self.user = user or os.getenv('USER', default='guest')
        self.secure = secure or False

    @property
    def url(self) -> str:
        return f'{self.host}:{self.port}'

    def channel(self) -> grpc.Channel:
        self._channel = grpc.aio.insecure_channel(target=self.url)
        return self._channel

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
