import os
import click
from pathlib import Path
from grpc_tools import protoc
from .echo import echo, green, magenta


@click.command(name='compile')
@click.option('-r', '--root', 'root',
              default="protos",
              type=str).
@click.option('-o', '--output', 'output',
              default=".",
              type=str)
def compile_command(root: str, output: str) -> None:
    echo.verbose(count=2, text=f'compile called with args: {root = } | {output = } ')
    proto_files = _get_all_proto_files(root=root)
    _mkdirs(root, proto_files)
    printed_protos = ",".join([f"\n\t{green(file) }"for file in proto_files])
    echo(f'Compiling proto files @ {green(root)}: [{printed_protos}\n]')
    command_args = [
        "grpc_tools.protoc",
        f"--proto_path={root}",
        f"--python_out={output}",
        f"--pyi_out={output}",
        f"--grpc_python_out={output}",
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


def _mkdirs(root: str, paths_to_proto_files: list[str]) -> None:
    paths_without_proto_root = [file_path.replace(root, ".") for file_path in paths_to_proto_files]
    echo.verbose(count=2, text=f'_mkdirs: {paths_without_proto_root = }')
    checked_paths: set[Path] = set()
    for file_path in paths_without_proto_root:
        dir_path = Path(os.path.join(*file_path.split(os.sep)[:-1]))
        if dir_path not in checked_paths:
            echo.verbose(f'Checking for directory @ {magenta(str(dir_path))}')
            dir_path.mkdir(parents=True, exist_ok=True)
            checked_paths.add(dir_path)


if __name__ == '__main__':
  compile_command()

