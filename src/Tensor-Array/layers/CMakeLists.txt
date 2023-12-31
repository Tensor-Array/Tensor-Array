cmake_minimum_required(VERSION 3.0.0)

project(TensorCore_Layers VERSION 0.1.0 LANGUAGES C CXX)

file(GLOB tensor_core_src "*.hh" "*.cc")

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
get_filename_component(PARENT_PARENT_DIR ${PARENT_DIR} DIRECTORY)


add_library(TensorCore_Layers SHARED ${tensor_core_src})

target_include_directories(TensorCore_Layers PRIVATE ${PARENT_PARENT_DIR})
target_link_libraries(TensorCore_Layers TensorCore_Core)
