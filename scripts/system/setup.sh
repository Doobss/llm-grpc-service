#!/bin/bash

CUDA_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv | grep -E '[0-9]{1,4}')"
echo "CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}"
LOCAL_NVCC__VERSION="$(nvcc --version &> /dev/null)"
export CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"


if [[ -z "${LOCAL_NVCC__VERSION}" ]]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    # export PATH="/usr/local/cuda/bin:${PATH}"
    LOCAL_NVCC__VERSION="$(nvcc --version &> /dev/null)"
fi

echo "Using nvcc version=${LOCAL_NVCC__VERSION}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

sudo apt install pkg-config libssl-dev protobuf-compiler -y

export PKG_CONFIG_PATH=/opt/miniconda/lib/pkgconfig


    # println!("cargo:rerun-if-changed=build.rs");
    # println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    # println!("cargo:rerun-if-env-changed=CUDA_PATH");
    # println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    # #[cfg(not(any(
    #     feature = "cuda-version-from-build-system",
    #     feature = "cuda-12040",
    #     feature = "cuda-12030",
    #     feature = "cuda-12020",
    #     feature = "cuda-12010",
    #     feature = "cuda-12000",
    #     feature = "cuda-11080",
    #     feature = "cuda-11070",
    # )))]
    # compile_error!("Must specify one of the following features: [cuda-version-from-build-system, cuda-12040, cuda-12030, cuda-12020, cuda-12010, cuda-12000, cuda-11080, cuda-11070]");

    # #[cfg(feature = "cuda-version-from-build-system")]
    # cuda_version_from_build_system();
