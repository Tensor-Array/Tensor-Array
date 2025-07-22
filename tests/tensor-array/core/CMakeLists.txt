cmake_minimum_required(VERSION 3.18)

set(
    TensorArray_tests_src
    "tensor_array_test.cc"
    "print_output.cc"
    "gradient.cc")

enable_testing()

create_test_sourcelist(
    TensorArray_tests
    "test_driver.cc"
    ${TensorArray_tests_src})

add_executable(tensorarray_core_tests ${TensorArray_tests})
target_include_directories(tensorarray_core_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_core_tests TensorArray::Core)

foreach(test ${TensorArray_tests_src})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND tensorarray_core_tests ${TName})
endforeach()
