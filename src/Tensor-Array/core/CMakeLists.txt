cmake_minimum_required(VERSION 3.0.0)
project(TensorCore_Core VERSION 0.1.0 LANGUAGES CUDA C CXX)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

file(GLOB tensor_core_src "*.hh" "*.cc" "*.cu")

add_library(TensorCore_Core SHARED ${tensor_core_src})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS TensorCore_Core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
