cmake_minimum_required(VERSION 3.8.0)

enable_language(C CXX)

file(GLOB TensorArray_src "*.cc")
file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor_array/layers
    COMPONENT headers)

add_library(TensorArray::Layers SHARED ${TensorArray_src})

target_include_directories(TensorArray::Layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(TensorArray::Layers TensorArray::Core)

install(
    TARGETS TensorArray::Layers
    EXPORT TensorArrayTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries)
