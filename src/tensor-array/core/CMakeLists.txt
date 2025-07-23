cmake_minimum_required(VERSION 3.18)

file(GLOB TensorArray_inc "*.hh")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor-array/core
    COMPONENT headers)

find_package(hip)
find_package(CUDAToolkit)

if (hip_FOUND)
    include("${PROJECT_SOURCE_DIR}/cmake/gpu-compiler/add-rocm-hip.cmake")
    add_for_ROCm(tensorarray_core)
elif(CUDAToolkit_FOUND)
    include("${PROJECT_SOURCE_DIR}/cmake/gpu-compiler/add-cuda-nvcc.cmake")
    add_for_CUDA(tensorarray_core)
else()
    file(GLOB TensorArray_src_cc "*.cc")
    add_library(tensorarray_core SHARED ${TensorArray_src_cc})
endif()

install(
    TARGETS tensorarray_core
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
    COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/core
    COMPONENT Development)

add_library(TensorArray::Core ALIAS tensorarray_core)
