
set(
    TensorArray_tests_src
    "tests/tensor-array/core/tensor_array_test.cc"
    "tests/tensor-array/core/print_output.cc"
    # "tests/tensor-array/core/tensor_operators.cc"
    # "tests/tensor-array/core/tensor_matmul_transpose.cc"
    # "tests/tensor-array/core/gradient.cc"
    )

enable_testing()

create_test_sourcelist(
    TensorArray_tests
    "tests/tensor-array/core/test_driver.cc"
    ${TensorArray_tests_src})

add_executable(tensorarray_core_tests ${TensorArray_tests})
target_include_directories(tensorarray_core_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_core_tests TensorArray::Core)

foreach(test ${TensorArray_tests_src})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND tensorarray_core_tests ${TName})
endforeach()
