cmake_minimum_required(VERSION 3.18)

file(GLOB TensorArray_src "*.cc")
file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor-array/layers
    COMPONENT headers)

add_library(tensorarray_layers SHARED ${TensorArray_src})

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
