set(TensorArray_Layers_Dir tensor-array/layers)

file(
    GLOB TensorArray_Layers_inc
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Layers_Dir}/*.h"
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Layers_Dir}/*.hh"
)

install(
    FILES ${TensorArray_Layers_inc}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${TensorArray_Layers_Dir}"
    COMPONENT headers
)

file(GLOB TensorArray_Layers_src "${PROJECT_SOURCE_DIR}/src/${TensorArray_Layers_Dir}/*.cc")

add_library(tensorarray_layers SHARED ${TensorArray_Layers_src})

target_include_directories(tensorarray_layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_layers TensorArray::Core)

set_property(TARGET tensorarray_layers PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_layers PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_layers PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_layers PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_layers PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_layers PROPERTY CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_definitions(tensorarray_layers PRIVATE TENSOR_ARRAY_LAYERS_EXPORTS)
endif()

install(
    TARGETS tensorarray_layers
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/tensor-array"
    COMPONENT Runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TensorArray_Layers_Dir}"
    COMPONENT Development
)

add_library(TensorArray::Layers ALIAS tensorarray_layers)
