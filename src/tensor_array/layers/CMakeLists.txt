cmake_minimum_required(VERSION 3.8.0)

enable_language(C CXX)

file(GLOB tensor_core_src "*.hh" "*.cc")

add_library(TensorArray_Layers SHARED ${tensor_core_src})

target_include_directories(TensorArray_Layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(TensorArray_Layers PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(TensorArray_Layers -lTensorArray_Core)

install(TARGETS TensorArray_Layers LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)