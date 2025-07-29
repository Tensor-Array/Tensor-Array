ROCM_PACKAGES_IN=(
    rocm-hip-runtime-devel
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

LINUX_CODENAME=$(lsb_release -cs)
LINUX_CODENAME="${LINUX_CODENAME,,}"

LOCATION_TEMP=${temp}

ROCM_VERSION_MAJOR_MINOR=${rocm}

CPU_ARCH=$(uname -m)
if [[ "${CPU_ARCH}" == "aarch64" ]]
then
    CPU_ARCH="sbsa"
fi


ROCM_PACKAGES=""
for package in "${ROCM_PACKAGES_IN[@]}"
do : 
    # Build the full package name and append to the string.
    ROCM_PACKAGES+=" ${package}"
done
echo "ROCM_PACKAGES ${ROCM_PACKAGES}"

GPG_FILENAME="rocm.gpg.key"
GPG_URL="https://repo.radeon.com/rocm/${GPG_FILENAME}"
REPO_URL="https://repo.radeon.com/rocm/apt/${rocm}/"

is_root=false
if (( $EUID == 0))
then
   is_root=true
fi
# Find if sudo is available
has_sudo=false
if command -v sudo &> /dev/null
then
    has_sudo=true
fi
# Decide if we can proceed or not (root or sudo is required) and if so store whether sudo should be used or not. 
if [ "$is_root" = false ] && [ "$has_sudo" = false ]
then 
    echo "Root or sudo is required. Aborting."
    exit 1
elif [ "$is_root" = false ]
then
    USE_SUDO=sudo
else
    USE_SUDO=
fi

KEYRINGS_DIR=/etc/apt/keyrings

if [ ! -e $KEYRINGS_DIR ]
then
    echo "Create directory: ${KEYRINGS_DIR}"
    $USE_SUDO mkdir --parents --mode=0755 ${KEYRINGS_DIR}
fi

ROCM_GPG_KEYRING=${KEYRINGS_DIR}/rocm.gpg

echo "Adding ROCm Repository:"
wget ${GPG_URL} -O - | \
    gpg --dearmor | $USE_SUDO tee ${ROCM_GPG_KEYRING} > /dev/null
echo "deb [arch=amd64 signed-by=${ROCM_GPG_KEYRING}] ${REPO_URL} ${LINUX_CODENAME} main" \
    | $USE_SUDO tee /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | $USE_SUDO tee /etc/apt/preferences.d/rocm-pin-600
echo "Adding ROCm Repository completed."
$USE_SUDO apt-get update

$USE_SUDO apt-get -y install ${ROCM_PACKAGES}

if [[ $? -ne 0 ]]
then
    echo "ROCm Installation Error."
    exit 1
fi

ROCM_PATH=/opt/rocm-${rocm}
echo "ROCM_PATH=${ROCM_PATH}"
export ROCM_PATH=${ROCM_PATH}
export PATH="$PATH:$ROCM_PATH/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ROCM_PATH/lib64"

if [[ $GITHUB_ACTIONS ]]
then
    echo "Adding ROCM to ROCM_PATH, PATH and LD_LIBRARY_PATH"
    echo "${ROCM_PATH}/bin" >> $GITHUB_PATH
    echo "ROCM_PATH=${ROCM_PATH}" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi
