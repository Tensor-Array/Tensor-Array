cmake_minimum_required(VERSION 3.8.0)

enable_language(CUDA C CXX)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

file(GLOB TensorArray_src "*.cc" "*.cu")
file(GLOB TensorArray_inc "*.hh")

# file(MAKE_DIRECTORY "include/tensor_array/core")

install(FILES ${TensorArray_inc}
DESTINATION include/tensor_array/core
COMPONENT headers)

add_library(TensorArray_Core SHARED ${TensorArray_src})
target_link_libraries(TensorArray_Core cublas)

install(
    TARGETS TensorArray_Core
    LIBRARY DESTINATION lib
    COMPONENT libraries
    ARCHIVE DESTINATION lib
    COMPONENT libraries)
