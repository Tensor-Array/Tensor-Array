set(TensorArray_Core_Dir tensor-array/core)

file(
    GLOB TensorArray_Core_inc
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Core_Dir}/*.h"
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Core_Dir}/*.hh"
)

install(
    FILES ${TensorArray_Core_inc}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${TensorArray_Core_Dir}"
    COMPONENT headers
)

include(CheckLanguage)
check_language(CUDA)

file(GLOB TensorArray_Core_cc "${PROJECT_SOURCE_DIR}/src/${TensorArray_Core_Dir}/*.cc")

if (CMAKE_CUDA_COMPILER)
    file(GLOB TensorArray_Core_cu "${PROJECT_SOURCE_DIR}/src/${TensorArray_Core_Dir}/*.cu")
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(tensorarray_core_object OBJECT ${TensorArray_Core_cc} ${TensorArray_Core_cu})
    set_property(TARGET tensorarray_core_object PROPERTY CUDA_STANDARD 17)
    set_property(TARGET tensorarray_core_object PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core_object PROPERTY CUDA_EXTENSIONS OFF)
    set_property(TARGET tensorarray_core_object PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(tensorarray_core_object PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDAToolkit_INCLUDE_DIRS}>)
    target_link_libraries(tensorarray_core_object PRIVATE $<$<LINK_LANGUAGE:C,CXX>:CUDA::cublas>)
    if(MSVC)
        target_compile_definitions(tensorarray_core_object PRIVATE TENSOR_ARRAY_CORE_EXPORTS)
    endif()
        
        # find_package(CUDAToolkit REQUIRED)
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
else()
    add_library(tensorarray_core_object OBJECT ${TensorArray_Core_cc})
endif()

# file(MAKE_DIRECTORY "include/tensor_array/core")

set_property(TARGET tensorarray_core_object PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_core_object PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core_object PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_core_object PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_core_object PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core_object PROPERTY CXX_EXTENSIONS OFF)

# shared libraries need PIC
set_property(TARGET tensorarray_core_object PROPERTY POSITION_INDEPENDENT_CODE 1)

# shared and static libraries built from the same object files
add_library(tensorarray_core SHARED $<TARGET_OBJECTS:tensorarray_core_object>)
add_library(tensorarray_core_static STATIC $<TARGET_OBJECTS:tensorarray_core_object>)

install(
    TARGETS tensorarray_core
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/tensor-array"
    COMPONENT Runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TensorArray_Core_Dir}"
    COMPONENT Development
)

install(
    TARGETS tensorarray_core_static
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/tensor-array"
    COMPONENT Runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TensorArray_Core_Dir}"
    COMPONENT Development
)

add_library(TensorArray::core ALIAS tensorarray_core)
add_library(TensorArray::core_static ALIAS tensorarray_core_static)
add_library(TensorArray::core_object ALIAS tensorarray_core_object)
