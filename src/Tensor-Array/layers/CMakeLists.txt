cmake_minimum_required(VERSION 3.8.0)

enable_language(C CXX)

file(GLOB tensor_core_src "*.hh" "*.cc")

add_library(TensorCore_Layers SHARED ${tensor_core_src})

target_include_directories(TensorCore_Layers PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(TensorCore_Layers PRIVATE ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(TensorCore_Layers -lTensorCore_Core)

install(TARGETS TensorCore_Layers LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
