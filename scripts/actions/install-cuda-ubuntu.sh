CUDA_PACKAGES_IN=(
    "cuda-compiler"
    "cuda-cudart"
    "cuda-nvtx"
    "cuda-nvrtc"
    "libcurand"
    "libcublas"
    "libcufft"
    "cuda-cccl"
)

LINUX_ID=$(lsb_release -sr)
LINUX_ID="${LINUX_ID}" | tr '[:upper:]' '[:lower:]'

LINUX_VERSION=$(lsb_release -sr)
LINUX_VERSION="${LINUX_VERSION//.}"

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
PIN_FILENAME="cuda-${LINUX_ID}${LINUX_VERSION}.pin"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/${LINUX_ID}${LINUX_VERSION}/${CPU_ARCH}/${PIN_FILENAME}"
KERYRING_PACKAGE_FILENAME="cuda-keyring_1.0-1_all.deb"
KEYRING_PACKAGE_URL="https://developer.download.nvidia.com/compute/cuda/repos/${LINUX_ID}${LINUX_VERSION}/${CPU_ARCH}/${KERYRING_PACKAGE_FILENAME}"
REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${LINUX_ID}${LINUX_VERSION}/${CPU_ARCH}/"

echo "Adding CUDA Repository"
wget ${PIN_URL}
$USE_SUDO mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
wget ${KEYRING_PACKAGE_URL} && ${USE_SUDO} dpkg -i ${KERYRING_PACKAGE_FILENAME} && rm ${KERYRING_PACKAGE_FILENAME}
$USE_SUDO add-apt-repository "deb ${REPO_URL} /"
$USE_SUDO apt-get update

echo "Installing CUDA packages ${CUDA_PACKAGES}"
if [ $(dpkg-query -W --showformat='${Status}\n' ${CUDA_PACKAGES} | grep -c "install ok installed") -eq 0 ];
then
$USE_SUDO apt-get -y install ${ CUDA_PACKAGES }
fi

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi

mkdir -p ${ LOCATION_TEMP }/package-cache/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
$USE_SUDO dpkg -L ${ CUDA_PACKAGES } | while IFS= read -r f; do if test -f $f; then echo $f; fi; done | xargs cp --parents --target-directory ${ LOCATION_TEMP }}/package-cache/cuda-${CUDA_MAJOR}.${CUDA_MINOR}

CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding CUDA to CUDA_PATH, PATH and LD_LIBRARY_PATH"
    echo "${CUDA_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi