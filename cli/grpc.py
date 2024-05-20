import click
import service as grpc
from .echo import echo, grey, green


@click.command(name='grpc')
@click.option('-u', '--user', 'username',
              prompt="Please enter a user name",
              type=str,
              envvar='USER')
@click.option('-h', '--host', 'host',
              default="0.0.0.0",
              type=str,
              show_default=True,
              help="IP or domain name where the websocket service is hosted.")
@click.option('-p', '--port', 'port',
              default=8765,
              type=int,
              show_default=True,
              help="Port the socket_server service is listing to.")
def grpc_server_command(
        username: str,
        host: str,
        port: int,
    ) -> None:
    """
    Starts the llm llm server...
    """
    echo.verbose(f'''\n{green('Starting gprc server with the following params')}:''')
    echo.verbose(f'\t{username = } {grey("(overwrite with -U option)")}'
                 f'\n\t{host = } {grey("(overwrite with -H option)")}'
                 f'\n\t{port = } {grey("(overwrite with -P option)")}')
    grpc.start()


