#!/bin/bash

set -e
set -x

CUDA_PACKAGES_IN=(
    "nvcc"
    "cudart"
    "nvtx"
    "nvrtc"
    "thrust"
    "curand_dev"
    "cublas_dev"
    "cufft_dev"
)

CUDA_VERSION_MAJOR_MINOR=${cuda}

CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)

CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"
do :
    # Build the full package name and append to the string.
    CUDA_PACKAGES+=" ${package}_${CUDA_MAJOR}.${CUDA_MINOR}"
done
echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

CUDA_ROOT = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_MAJOR}.${CUDA_MINOR}"

curl --netrc-optional -L -nv -o cuda_installer.exe "https://developer.download.nvidia.com/compute/cuda/${cuda}/network_installers/cuda_${cuda}_windows_network.exe"
./cuda_installer.exe -s ${CUDA_PACKAGES}
rm -f cuda_installer.exe
