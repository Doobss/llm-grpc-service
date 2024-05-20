## Setup
```bash
sudo apt install python3.11 python3.11-venv -y
python3.11 -m venv venv
source venv/bin/activate
python3.11 -m pip install .
llm complie # complies protos (generates python classes)
python3.11 -m pip install .[server]
```

## Compile protos

```bash
 python3 -m grpc_tools.protoc \
 -I protos \
 --python_out=.  \
 --pyi_out=. \
 --grpc_python_out=. \
 ./protos/[path to proto dir]/*.proto
```