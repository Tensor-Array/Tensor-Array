CUDA_PACKAGES_IN=(
    "cuda-compiler"
    "cuda-cudart"
    "cuda-nvtx"
    "cuda-nvrtc"
    "cuda-cccl"
    "libcurand-devel"
    "libcublas-devel"
    "libcufft-devel"
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


LINUX_ID=$(lsb_release -si)
LINUX_ID="${LINUX_ID,,}"

LINUX_VERSION=$(lsb_release -sr)
LINUX_VERSION="${LINUX_VERSION//.}"

LINUX_VERSION_MAJOR_MINOR=$(lsb_release -sr)
LINUX_MAJOR=$(echo "${LINUX_VERSION_MAJOR_MINOR}" | cut -d. -f1)
LINUX_MINOR=$(echo "${LINUX_VERSION_MAJOR_MINOR}" | cut -d. -f2)
LINUX_PATCH=$(echo "${LINUX_VERSION_MAJOR_MINOR}" | cut -d. -f3)

if [[ "${LINUX_ID}" == "almalinux" || "${LINUX_ID}" == "centos" || "${LINUX_ID}" == "oracle" ]]; then
    echo "LINUX_ID: ${LINUX_ID} change to rhel"
    LINUX_ID="rhel"
    LINUX_VERSION=${LINUX_MAJOR}
fi

LOCATION_TEMP=${temp}

CUDA_VERSION_MAJOR_MINOR=${cuda}

CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)

CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"
do : 
    # @todo This is not perfect. Should probably provide a separate list for diff versions
    # cuda-compiler-X-Y if CUDA >= 9.1 else cuda-nvcc-X-Y
    if [[ "${package}" == "cuda-nvcc" ]] && version_ge "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="cuda-compiler"
    elif [[ "${package}" == "cuda-compiler" ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="cuda-nvcc"
    # CUB/Thrust  are packages in cuda-thrust in 11.3, but cuda-cccl in 11.4+
    elif [[ "${package}" == "cuda-thrust" || "${package}" == "cuda-cccl" ]]; then
        # CUDA cuda-thrust >= 11.4
        if version_ge "$CUDA_VERSION_MAJOR_MINOR" "11.4" ; then
            package="cuda-cccl"
        # Use cuda-thrust > 11.2
        elif version_ge "$CUDA_VERSION_MAJOR_MINOR" "11.3" ; then
            package="cuda-thrust"
        # Do not include this pacakge < 11.3
        else
            continue
        fi
    fi
    # Build the full package name and append to the string.
    CUDA_PACKAGES+=" ${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
done
echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

CPU_ARCH="x86_64"
REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${LINUX_ID}${LINUX_VERSION}/${CPU_ARCH}/cuda-${LINUX_ID}${LINUX_VERSION}.repo"

is_root=false
if (( $EUID == 0)); then
   is_root=true
fi
# Find if sudo is available
has_sudo=false
if command -v sudo &> /dev/null ; then
    has_sudo=true
fi
# Decide if we can proceed or not (root or sudo is required) and if so store whether sudo should be used or not. 
if [ "$is_root" = false ] && [ "$has_sudo" = false ]; then 
    echo "Root or sudo is required. Aborting."
    exit 1
elif [ "$is_root" = false ] ; then
    USE_SUDO=sudo
else
    USE_SUDO=
fi

echo "Adding CUDA Repository"
$USE_SUDO dnf config-manager --add-repo ${REPO_URL}
$USE_SUDO dnf clean all

$USE_SUDO dnf -y install ${CUDA_PACKAGES}

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi

CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATH/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATH/lib64"

if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding CUDA to CUDA_PATH, PATH and LD_LIBRARY_PATH"
    echo "${CUDA_PATH}" >> $GITHUB_PATH
    echo "PATH=$PATH:$CUDA_PATH/bin" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_PATH}/lib" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64" >> $GITHUB_ENV
fi
