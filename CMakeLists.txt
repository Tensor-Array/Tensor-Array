cmake_minimum_required(VERSION 3.18)

project(TensorArray)

include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

add_subdirectory("src/tensor_array/core")
add_subdirectory("src/tensor_array/layers")

set(CPACK_PACKAGE_NAME "TensorArray")
set(CPACK_PACKAGE_VENDOR "TensorArray-Creators")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY)

set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "A machine learning libraries")

install(
    EXPORT TensorArrayTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorArray
    NAMESPACE TensorArray::
    FILE TensorArrayTargets.cmake
    )

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "Config.cmake.in"
    "TensorArrayConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorArray
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
    )

### Install Config and ConfigVersion files
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorArrayConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorArray
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
