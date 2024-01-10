cmake_minimum_required(VERSION 3.8.0)

enable_language(C CXX)

file(GLOB TensorArray_src "*.cc")
file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION "include/tensor_array/layers"
    COMPONENT headers)

add_library(tensorarray_layers SHARED ${TensorArray_src})
add_library(TensorArray::Layers ALIAS tensorarray_layers)

target_include_directories(tensorarray_layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(tensorarray_layers PRIVATE "${PROJECT_SOURCE_DIR}/build-temp/library")
target_link_libraries(tensorarray_layers TensorArray::Core)

install(
    TARGETS tensorarray_layers
    LIBRARY DESTINATION lib
    COMPONENT libraries
    ARCHIVE DESTINATION lib
    COMPONENT libraries)
