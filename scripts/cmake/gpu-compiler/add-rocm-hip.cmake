function(add_for_ROCm TA_TARGET)
    enable_language(HIP)
    # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
    # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
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

    set_property(TARGET ${TA_TARGET} PROPERTY HIP_STANDARD 17)
    set_property(TARGET ${TA_TARGET} PROPERTY HIP_STANDARD_REQUIRED ON)
    set_property(TARGET ${TA_TARGET} PROPERTY HIP_EXTENSIONS OFF)

    foreach(TensorArray_src_hip ${TensorArray_src_cu})
        set_source_files_properties(${TensorArray_src_hip} PROPERTIES LANGUAGE HIP)
    endforeach()

    target_link_libraries(${TA_TARGET} PRIVATE hip::host hip::device)
endfunction()
