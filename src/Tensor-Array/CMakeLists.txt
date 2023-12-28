cmake_minimum_required(VERSION 3.0.0)

file(GLOB tensor_core_src "layers/*.hh" "layers/*.cc" "core/*.hh" "core/*.cc" "core/*.cu")

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)


add_library(TensorCore SHARED ${tensor_core_src})

target_include_directories(TensorCore PRIVATE ${PARENT_DIR})
