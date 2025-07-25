cmake_minimum_required(VERSION 3.18)

project(TensorArray)

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

block(SCOPE_FOR POLICIES)
    include(CheckLanguage)

    check_language(HIP)
    check_language(CUDA)
    file(GLOB TensorArray_Core_cc "${PROJECT_SOURCE_DIR}/src/tensor-array/core/*.cc")

    if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
        file(GLOB TensorArray_Core_cu "${PROJECT_SOURCE_DIR}/src/tensor-array/core/*.cu")
    endif()

    if(CMAKE_HIP_COMPILER)
        block(PROPAGATE tensorarray_core)
            enable_language(HIP)
            find_package(hip REQUIRED)

            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
            set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD 17)
            set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD_REQUIRED ON)
            set_property(TARGET tensorarray_core PROPERTY HIP_EXTENSIONS OFF)

            foreach(TensorArray_src_hip ${TensorArray_src_cu})
                set_source_files_properties(${TensorArray_src_hip} PROPERTIES LANGUAGE HIP)
            endforeach()

            target_link_libraries(tensorarray_core PRIVATE hip::host hip::device)
        endblock()
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    elif(CMAKE_CUDA_COMPILER)
        block(PROPAGATE tensorarray_core)
            enable_language(CUDA)
            if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
                message(NOTICE "cuda include path: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
            else()
                message(FATAL_ERROR "No cuda include path")
            endif()
            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
            set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD 17)
            set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD_REQUIRED ON)
            set_property(TARGET tensorarray_core PROPERTY CUDA_EXTENSIONS OFF)
            set_property(TARGET tensorarray_core PROPERTY CMAKE_CUDA_SEPARABLE_COMPILATION ON)
            target_include_directories(tensorarray_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endblock()
        
        # find_package(CUDAToolkit REQUIRED)
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    else()
        block(PROPAGATE tensorarray_core TensorArray_Core_cc TensorArray_Core_cu)
            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
        endblock()
    endif()


    # file(MAKE_DIRECTORY "include/tensor_array/core")

    set_property(TARGET tensorarray_core PROPERTY C_STANDARD 11)
    set_property(TARGET tensorarray_core PROPERTY C_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core PROPERTY C_EXTENSIONS OFF)

    set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD 17)
    set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core PROPERTY CXX_EXTENSIONS OFF)

    install(
        TARGETS tensorarray_core
        EXPORT TensorArrayTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
        COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/core
        COMPONENT Development)

    add_library(TensorArray::Core ALIAS tensorarray_core)

endblock()

block(SCOPE_FOR POLICIES)
    file(GLOB TensorArray_Layers_src "${PROJECT_SOURCE_DIR}/src/tensor-array/layers/*.cc")

    add_library(tensorarray_layers SHARED ${TensorArray_Layers_src})

    target_include_directories(tensorarray_layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(tensorarray_layers TensorArray::Core)

    set_property(TARGET tensorarray_layers PROPERTY C_STANDARD 11)
    set_property(TARGET tensorarray_layers PROPERTY C_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_layers PROPERTY C_EXTENSIONS OFF)

    set_property(TARGET tensorarray_layers PROPERTY CXX_STANDARD 17)
    set_property(TARGET tensorarray_layers PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_layers PROPERTY CXX_EXTENSIONS OFF)

    install(
        TARGETS tensorarray_layers
        EXPORT TensorArrayTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
        COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/layers
        COMPONENT Development)

    add_library(TensorArray::Layers ALIAS tensorarray_layers)
endblock()

block(SCOPE_FOR POLICIES)
    file(
        GLOB TensorArray_Interpreter_src
        "${PROJECT_SOURCE_DIR}/src/tensor-array/interp/*.c"
        "${PROJECT_SOURCE_DIR}/src/tensor-array/interp/*.cc"
    )
    add_executable(tensorarray_interpreter ${TensorArray_Interpreter_src})

    target_include_directories(tensorarray_interpreter PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(tensorarray_interpreter TensorArray::Core)

    set_property(TARGET tensorarray_interpreter PROPERTY C_STANDARD 11)
    set_property(TARGET tensorarray_interpreter PROPERTY C_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_interpreter PROPERTY C_EXTENSIONS OFF)

    set_property(TARGET tensorarray_interpreter PROPERTY CXX_STANDARD 17)
    set_property(TARGET tensorarray_interpreter PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_interpreter PROPERTY CXX_EXTENSIONS OFF)

    install(
        TARGETS tensorarray_interpreter
        EXPORT TensorArrayTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
        COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/interp
        COMPONENT Development)
    #[[
    add_custom_command(
        OUTPUT test.tmp
        DEPENDS tensorarray_interpreter
        POST_BUILD
        COMMAND tensorarray_interpreter)
    ]]
    add_executable(TensorArray::Interpreter ALIAS tensorarray_interpreter)

endblock()

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
