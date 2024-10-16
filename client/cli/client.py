import click
import asyncio
from typing import Union
from uuid import uuid4

from reflection import LLmClient, GrpcReflection
from reflection.promptClient import PromptClient
from .echo import echo, grey, green, red, magenta, cyan



@click.command(name='client')
@click.option('-u', '--user', 'username',
              prompt="Please enter a user name",
              type=str,
              envvar='USER')
@click.option('-h', '--host', 'host',
              default="127.0.0.1",
              type=str,
              show_default=True,
              help="IP or domain name where the websocket service is hosted.")
@click.option('-p', '--port', 'port',
              default=50051,
              type=int,
              show_default=True,
              help="Port the socket_server service is listing to.")
@click.option('-s', '--secure', 'secure',
              default=False,
              type=bool,
              show_default=True,
              help="wss vs ws.")
@click.option('-t', '--template', 'template',
              default=True,
              type=bool,
              show_default=True,
              help="Whether or not to auto apply the models template.")
def client_command(
        username: str,
        host: str,
        port: int,
        secure: bool,
        template: bool
    ) -> None:
    """
    Opens a connection to the llm and starts a simple chat session.
    """
    echo.verbose(f'''\n{green('Starting client with the following params')}:''')
    echo.verbose(f'\t{username = } {grey("(overwrite with -U option)")}'
                 f'\n\t{host = } {grey("(overwrite with -H option)")}'
                 f'\n\t{port = } {grey("(overwrite with -P option)")}'
                 f'\n\t{secure = } {grey("(overwrite with -S option)")}'
                 f'\n\t{template = } {grey("(overwrite with -T option)")}\n')
    asyncio.run(start_client(username, host, port, secure, template))


async def start_client(
        username: str,
        host: str,
        port: int,
        secure: bool,
        template: bool
    ) -> None:
    connect_attempts = 0
    max_retries = 10

    target = f'{host}:{port}'
    reflection = GrpcReflection(target=target)
    prompt_client = PromptClient(reflection=reflection)
    llm_client = LLmClient(reflection=reflection)
    
    messages: list[dict] = [

    ]
    def add_message(content: str, role: str) -> None:
        messages.append({ "role": role, "content": content })

    def add_user_message(content: str) -> None:
        add_message(content, 'user')

    def add_assistant_message(content: str) -> None:
        add_message(content, 'assistant')

    
    def apply_template(messages: list[dict]) -> str:
        response = prompt_client.apply_template(messages=messages)
        if hasattr(response, 'content') and isinstance(response.content, str):
            return f'{response.content}'
        else:
            raise ValueError(f'Error applying template: {str(response)}')
        
    while max_retries > connect_attempts:
        try:
            connect_attempts = 0
            echo(f'''{green(f'Client connected')}''', nl=False)
            echo.verbose(f' @ {magenta(target)}', nl=False)
            echo("\n")
            while True:
                input_prompt = click.prompt(f'{green("user")}')
                add_user_message(input_prompt)
                last_message: Union[str, None] = None
                for message in llm_client.prompt(content=apply_template(messages=messages)):
                    last_message = echo_message(message, last_message)
                add_assistant_message(last_message)
        except click.exceptions.Abort as close_event:
            echo(f'\n{grey("disconnecting client")}')            
            break
        except Exception as error:
            echo(f'''\n{red("Error occurred")}: {str(error)}''', nl=False)
            echo.verbose(f''' {type(error)}''', nl=False)
            echo(f'''\n{grey('Attempting to reconnect')}''')
            connect_attempts += 1
    if connect_attempts >= max_retries:
        echo(f'''{red("Exiting")}: exceeded max connection retries, try again in a bit.''')


def echo_message(message, last_message: Union[str, None]) -> str:
    content = message.content
    if last_message is not None:
        new_content = content.replace(last_message, "")
        echo(new_content, nl=False)
    else:
        echo(f'{cyan("llm")}: {content}', nl=False)
    if message.is_end_of_sequence:
        extra_info = get_message_info(message)
        echo(f'\n\t{extra_info}')
    return message.content




def get_message_info(message) -> str:
    meta = message.meta if hasattr(message, 'meta') else None
    model = meta.model if meta is not None and hasattr(meta, 'model') else None
    if meta is not None and model:
        return grey(f"model: {meta.model.type} "
                              f"token/sec: {meta.tokens_per_second:.2f} | "
                              f"ave batch size: {meta.average_batch_size:.2f} | "
                              f"device: {model.device} (x {model.number_of_devices or 1})| "
                              f"dtype: {model.dtype}\n")
    else:
        return '\n'

