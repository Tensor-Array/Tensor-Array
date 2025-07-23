function(add_for_CUDA TA_TARGET)
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    file(GLOB TensorArray_src_cc "*.cc")
    file(GLOB TensorArray_src_cu "*.cu")
    add_library(${TA_TARGET} SHARED ${TensorArray_src_cc} ${TensorArray_src_cu})

    set_property(TARGET ${TA_TARGET} PROPERTY C_STANDARD 11)
    set_property(TARGET ${TA_TARGET} PROPERTY C_STANDARD_REQUIRED ON)
    set_property(TARGET ${TA_TARGET} PROPERTY C_EXTENSIONS OFF)

    set_property(TARGET ${TA_TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${TA_TARGET} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${TA_TARGET} PROPERTY CXX_EXTENSIONS OFF)

    set_property(TARGET ${TA_TARGET} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${TA_TARGET} PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET ${TA_TARGET} PROPERTY CUDA_EXTENSIONS OFF)

    target_link_libraries(
        ${TA_TARGET} PRIVATE
        CUDA::cudart CUDA::cudart_static
        CUDA::curand CUDA::curand_static
        CUDA::cublas CUDA::cublas_static
        )
endfunction()
