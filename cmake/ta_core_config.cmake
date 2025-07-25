block(SCOPE_FOR POLICIES)
    include(CheckLanguage)

    include(CheckLanguage)
    check_language(HIP)
    check_language(CUDA)

    file(GLOB TensorArray_Core_cc "${PROJECT_SOURCE_DIR}/src/tensor-array/core/*.cc")

    if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
        file(GLOB TensorArray_Core_cu "${PROJECT_SOURCE_DIR}/src/tensor-array/core/*.cu")
    endif()

    if(CMAKE_HIP_COMPILER)
        block(PROPAGATE tensorarray_core)
            enable_language(HIP)
            find_package(hip REQUIRED)

            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
            set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD 17)
            set_property(TARGET tensorarray_core PROPERTY HIP_STANDARD_REQUIRED ON)
            set_property(TARGET tensorarray_core PROPERTY HIP_EXTENSIONS OFF)

            foreach(TensorArray_src_hip ${TensorArray_src_cu})
                set_source_files_properties(${TensorArray_src_hip} PROPERTIES LANGUAGE HIP)
            endforeach()

            target_link_libraries(tensorarray_core PRIVATE hip::host hip::device)
        endblock()
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    elseif(CMAKE_CUDA_COMPILER)
        block(PROPAGATE tensorarray_core)
            enable_language(CUDA)
            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
            set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD 17)
            set_property(TARGET tensorarray_core PROPERTY CUDA_STANDARD_REQUIRED ON)
            set_property(TARGET tensorarray_core PROPERTY CUDA_EXTENSIONS OFF)
            set_property(TARGET tensorarray_core PROPERTY CMAKE_CUDA_SEPARABLE_COMPILATION ON)
            set_source_files_properties(data_type_wrapper.cc PROPERTIES LANGUAGE CUDA)
        endblock()
        
        # find_package(CUDAToolkit REQUIRED)
        # set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
        # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    else()
        block(PROPAGATE tensorarray_core TensorArray_Core_cc TensorArray_Core_cu)
            add_library(tensorarray_core SHARED ${TensorArray_Core_cc} ${TensorArray_Core_cu})
        endblock()
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
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
        COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/core
        COMPONENT Development)

    add_library(TensorArray::Core ALIAS tensorarray_core)

endblock()
