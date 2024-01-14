cmake_minimum_required(VERSION 3.8.0)

file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor_array/core
    COMPONENT headers)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
enable_language(CUDA C CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

file(GLOB TensorArray_src "*.cc" "*.cu")

# file(MAKE_DIRECTORY "include/tensor_array/core")

add_library(tensorarray_core SHARED ${TensorArray_src})
target_link_libraries(tensorarray_core PRIVATE CUDA::cublas)

endif(CUDAToolkit_FOUND)

add_library(TensorArray::Core ALIAS tensorarray_core)

install(
    TARGETS tensorarray_core
    EXPORT TensorArrayTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries)
