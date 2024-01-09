cmake_minimum_required(VERSION 3.8.0)

project(Tensor-Array)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

add_subdirectory("src/tensor_array/core")

# Create a custom target for installation
add_custom_target(install_target
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
    COMMENT "Installing before build"
    VERBATIM
)

# Add the custom installation target as a dependency for the default build target
add_custom_target(build_all
    DEPENDS install_target
)

add_subdirectory("src/tensor_array/layers")

set(CPACK_PACKAGE_NAME "Tensor-Array")
set(CPACK_PACKAGE_VENDOR "TensorArray-Creators")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY)

set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "A machine learning libraries")

include(CPack)
cpack_add_component(libraries
DISPLAY_NAME "Libraries"
GROUP Development)

cpack_add_component(
    headers
    DISPLAY_NAME "C++ Headers"
    GROUP Development)

cpack_add_component_group(
    Development
    EXPANDED
    PROJECT_DESCRIPTION
    "All of the tools you'll ever need to develop software")
