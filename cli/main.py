import click
from .client import client_command
from .grpc import grpc_server_command
from .echo import echo
from .compile import compile_command


@click.group(name='llm')
@click.option('-v', '--verbose', 'verbose',
              default=0,
              type=int,
              show_default=True,
              count=True,
              help="wss vs ws.")
def llm_cli(verbose: int) -> None:
    echo.set_verbose_level(verbose)


llm_cli.add_command(client_command)
llm_cli.add_command(grpc_server_command)
llm_cli.add_command(compile_command)

if __name__ == '__main__':
    llm_cli()