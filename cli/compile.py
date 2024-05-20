import os
import click
from grpc_tools import protoc
from .echo import echo, green


@click.command(name='compile')
@click.option('-p', '--path', 'path',
              default="protos",
              type=str)
def compile_command(path: str) -> None:
    proto_files = _get_all_proto_files(root=path)
    printed_protos = ",".join([f"\n\t{green(file) }"for file in proto_files])
    echo(f'Compiling proto files @ {green(path)}: [{printed_protos}\n]')
    command_args = [
        "grpc_tools.protoc",
        f"--proto_path={path}",
        "--python_out=.",
        "--pyi_out=.",
        "--grpc_python_out=.",
        *proto_files
    ]
    echo.verbose(f'Running with args {command_args = }')
    protoc.main(command_args)


def _get_all_proto_files(root: str) -> list[str]:
    proto_files: list[str] = []
    inclusion_root = os.path.relpath(root)
    for root, dirs, files in os.walk(inclusion_root):
        for filename in files:
            if filename.endswith(".proto"):
                proto_files.append(
                    os.path.relpath(str(os.path.join(root, filename)))
                )
        for child_dir in dirs:
            proto_files.extend(_get_all_proto_files(child_dir))
    return proto_files
