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

add_library(tensorarray_layers_object OBJECT ${TensorArray_Layers_src})

target_include_directories(tensorarray_layers_object PRIVATE ${PROJECT_SOURCE_DIR}/src)

set_property(TARGET tensorarray_layers_object PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_layers_object PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_layers_object PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_layers_object PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_layers_object PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_layers_object PROPERTY CXX_EXTENSIONS OFF)

# shared libraries need PIC
set_property(TARGET tensorarray_layers_object PROPERTY POSITION_INDEPENDENT_CODE 1)

if(MSVC)
    target_compile_definitions(tensorarray_layers_object PRIVATE TENSOR_ARRAY_LAYERS_EXPORTS)
endif()

# shared and static libraries built from the same object files
add_library(tensorarray_layers SHARED $<TARGET_OBJECTS:tensorarray_layers_object>)
add_library(tensorarray_layers_static STATIC $<TARGET_OBJECTS:tensorarray_layers_object>)

target_link_libraries(tensorarray_layers PUBLIC TensorArray::core)
target_link_libraries(tensorarray_layers_static PUBLIC TensorArray::core_static)

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

install(
    TARGETS tensorarray_layers_static
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/tensor-array"
    COMPONENT Runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TensorArray_Layers_Dir}"
    COMPONENT Development
)

add_library(TensorArray::layers ALIAS tensorarray_layers)
add_library(TensorArray::layers_static ALIAS tensorarray_layers_static)
add_library(TensorArray::layers_object ALIAS tensorarray_layers_object)
