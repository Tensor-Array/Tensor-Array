#! /bin/bash

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
    "visual_studio_integration"
)

function version_ge() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$2" ]
}
# returns 0 (true) if a > b
function version_gt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_ge $1 $2
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le $1 $2
}


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

# CUDA_ROOT = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_MAJOR}.${CUDA_MINOR}"

curl --netrc-optional -L -nv -o cuda_installer.exe \
"https://developer.download.nvidia.com/compute/cuda/${cuda}/network_installers/cuda_${cuda}_windows_network.exe"
./cuda_installer.exe -s ${CUDA_PACKAGES}
rm -f cuda_installer.exe

CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_MAJOR}.${CUDA_MINOR}"
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}

if [[ $GITHUB_ACTIONS ]]
then
    echo "Adding CUDA to CUDA_PATH"
    echo "${CUDA_PATH}/bin" >> $GITHUB_PATH
    echo "CUDA_PATH=${CUDA_PATH}" >> $GITHUB_ENV
fi

