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
    add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
    set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET tensorarray_core PROPERTY CUDA_EXTENSIONS OFF)
    set_property(TARGET tensorarray_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(tensorarray_core PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDAToolkit_INCLUDE_DIRS}>)
    target_link_libraries(tensorarray_core PRIVATE $<$<LINK_LANGUAGE:C,CXX>:CUDA::cublas>)
    if(MSVC)
        target_compile_definitions(tensorarray_core PRIVATE TENSOR_ARRAY_CORE_EXPORTS)
    endif()
        
        # find_package(CUDAToolkit REQUIRED)
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
else()
    add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
endif()


# file(MAKE_DIRECTORY "include/tensor_array/core")

set_property(TARGET tensorarray_core PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_core PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_core PROPERTY CXX_EXTENSIONS OFF)

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

add_library(TensorArray::Core ALIAS tensorarray_core)
