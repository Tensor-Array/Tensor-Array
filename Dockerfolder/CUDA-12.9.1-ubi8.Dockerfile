FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubi8

RUN dnf update -y
RUN dnf upgrade -y
RUN dnf install curl -y

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.27.9"

# Optionally install the cmake for vcpkg
COPY scripts/packages-install/reinstall-cmake-rhel.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake-rhel.sh && /tmp/reinstall-cmake-rhel.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake-rhel.sh


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

WORKDIR /main-project
COPY ./ tensor-array/

WORKDIR tensor-array/build

RUN cmake ..
RUN cmake --build .
RUN cmake --install .
RUN ctest

WORKDIR ..
