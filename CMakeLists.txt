cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0104 NEW)

project(TensorArray C CXX)

include(GNUInstallDirs)
# set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

if(MSVC)
    add_compile_definitions(TENSOR_ARRAY_EXPORTS)
endif()

file(
    GLOB_RECURSE TensorArray_inc
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.hh"
)

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT headers
)

include(cmake/ta_core_config.cmake)
include(cmake/ta_layers_config.cmake)
include(cmake/ta_interp_config.cmake)

# add_subdirectory("src/tensor-array/core")
# add_subdirectory("src/tensor-array/layers")
# add_subdirectory("src/tensor-array/interp")

include(CTest)
if(BUILD_TESTING)
    add_subdirectory("tests/tensor-array/core")
endif()

set(CPACK_PACKAGE_NAME "TensorArray")
set(CPACK_PACKAGE_VENDOR "TensorArray-Creators")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY)

set(CPACK_PACKAGE_VERSION "0.2.0")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "A machine learning libraries")

install(
    EXPORT TensorArrayTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/cmake
    NAMESPACE TensorArray::
    FILE TensorArrayTargets.cmake
    )

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "Config.cmake.in"
    "TensorArrayConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/cmake
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfigVersion.cmake
    VERSION 0.2.0
    COMPATIBILITY SameMajorVersion
    )

### Install Config and ConfigVersion files
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/cmake
    )

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
