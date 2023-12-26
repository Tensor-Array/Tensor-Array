file(GLOB tensor_core_SRC "*.hh" "*.cc" "core/*.hh" "core/*.cc" "*.cu")

add_library(TensorCore SHARED ${tensor_core_SRC})
