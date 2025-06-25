cmake_minimum_required(VERSION 3.10)

file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor_array/core
    COMPONENT headers)

enable_language(CUDA)

include(FindCUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")

file(GLOB TensorArray_src "*.cc" "*.cu")

# file(MAKE_DIRECTORY "include/tensor_array/core")

cuda_add_library(tensorarray_core ${TensorArray_src} SHARED)

cuda_add_cublas_to_target(tensorarray_core)

install(
    TARGETS tensorarray_core
    EXPORT TensorArrayTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries)

add_library(TensorArray::Core ALIAS tensorarray_core)
