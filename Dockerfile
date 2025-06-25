FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu20.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.27.8"

# Optionally install the cmake for vcpkg
COPY script/packages-install/reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

WORKDIR /tensor-array
COPY src/ ./src/
COPY CMakeLists.txt ./
COPY Config.cmake.in ./
WORKDIR /tensor-array

WORKDIR /tensor-array/build

RUN cmake ..
RUN make install

WORKDIR /tensor-array
