cmake_minimum_required(VERSION 3.0.0)

file(GLOB tensor_core_SRC "layers/*.hh" "layers/*.cc" "core/*.hh" "core/*.cc" "*.cu")

add_library(TensorCore SHARED ${tensor_core_SRC})

target_include_directories(TensorCore PRIVATE ${CMAKE_CURRENT_LIST_DIR})
