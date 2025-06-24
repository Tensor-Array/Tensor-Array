FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get upgrade -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install cmake


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
