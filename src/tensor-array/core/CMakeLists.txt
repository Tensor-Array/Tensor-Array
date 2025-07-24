cmake_minimum_required(VERSION 3.18)

file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor-array/core
    COMPONENT headers)

include(CheckLanguage)

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    find_package(hip)
    # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
    # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
endif()

file(GLOB TensorArray_src_cc "*.cc")

if (USE_CUDA OR USE_ROCM_HIP)
    file(GLOB TensorArray_src_cu "*.cu")
endif()

# file(MAKE_DIRECTORY "include/tensor_array/core")

add_library(tensorarray_core SHARED ${TensorArray_src_cc} ${TensorArray_src_cu})

set_property(TARGET tensorarray_core PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_core PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core PROPERTY CXX_EXTENSIONS OFF)

if(CMAKE_HIP_COMPILER)
    set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD 17)
    set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core PROPERTY HIP_EXTENSIONS OFF)

    foreach(TensorArray_src_hip ${TensorArray_src_cu})
        set_source_files_properties(${TensorArray_src_hip} PROPERTIES LANGUAGE HIP)
    endforeach()

    target_link_libraries(tensorarray_core PRIVATE hip::host hip::device)
endif()

if (CMAKE_CUDA_COMPILER)
    set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core PROPERTY CUDA_EXTENSIONS OFF)
endif()

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
