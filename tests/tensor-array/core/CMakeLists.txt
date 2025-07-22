cmake_minimum_required(VERSION 3.18)

file(GLOB TensorArray_tests_src "*.cc")

enable_testing()

create_test_sourcelist(
    TensorArray_tests
    test_driver.cc
    ${TensorArray_tests_src})

add_executable(tensorarray_tests ${TensorArray_tests})
target_include_directories(tensorarray_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_tests TensorArray::Core)

foreach(test ${TensorArray_tests_src})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND tensorarray_tests ${TName})
endforeach()
