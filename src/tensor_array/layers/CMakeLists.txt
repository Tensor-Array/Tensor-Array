cmake_minimum_required(VERSION 3.8.0)

enable_language(C CXX)

file(GLOB TensorArray_src "*.cc")
file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION "include/tensor_array/layers"
    COMPONENT headers)

add_library(TensorArray_Layers SHARED ${TensorArray_src})

target_include_directories(TensorArray_Layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(TensorArray_Layers PRIVATE "${PROJECT_SOURCE_DIR}/build-temp/library")
target_link_libraries(TensorArray_Layers -lTensorArray_Core)

install(
    TARGETS TensorArray_Layers
    LIBRARY DESTINATION lib
    COMPONENT libraries
    ARCHIVE DESTINATION lib
    COMPONENT libraries)
