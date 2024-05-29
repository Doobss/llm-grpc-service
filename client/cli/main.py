import click
from .client import client_command
from .echo import echo


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
start = llm_cli


if __name__ == '__main__':
    llm_cli()