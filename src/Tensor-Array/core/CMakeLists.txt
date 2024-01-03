cmake_minimum_required(VERSION 3.8.0)
enable_language(CUDA C CXX)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

file(GLOB tensor_core_src "*.hh" "*.cc" "*.cu")

add_library(TensorCore_Core SHARED ${tensor_core_src})
target_link_libraries(TensorCore_Core cublas)

install(TARGETS TensorCore_Core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
