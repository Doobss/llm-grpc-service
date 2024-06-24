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

## System installs

Will need Cuda (version 12.4)
Cudnn

nccl
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev
```