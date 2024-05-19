from . import grpc


def start() -> None:
    grpc.server.start()


if __name__ == '__main__':
    start()
