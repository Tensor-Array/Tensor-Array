set(TensorArray_Interpreter_Dir tensor-array/interp)

file(
    GLOB TensorArray_Interpreter_inc
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Interpreter_Dir}/*.h"
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Interpreter_Dir}/*.hh"
)

install(
    FILES ${TensorArray_Interpreter_inc}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${TensorArray_Interpreter_Dir}"
    COMPONENT headers
)

file(
    GLOB TensorArray_Interpreter_src
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Interpreter_Dir}/*.c"
    "${PROJECT_SOURCE_DIR}/src/${TensorArray_Interpreter_Dir}/*.cc"
)
add_executable(tensorarray_interpreter ${TensorArray_Interpreter_src})

target_include_directories(tensorarray_interpreter PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_interpreter TensorArray::Core)

set_property(TARGET tensorarray_interpreter PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_interpreter PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_interpreter PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_interpreter PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_interpreter PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_interpreter PROPERTY CXX_EXTENSIONS OFF)

install(
    TARGETS tensorarray_interpreter
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/tensor-array"
    COMPONENT Runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TensorArray_Interpreter_Dir}"
    COMPONENT Development
)
    #[[
    add_custom_command(
        OUTPUT test.tmp
        DEPENDS tensorarray_interpreter
        POST_BUILD
        COMMAND tensorarray_interpreter)
    ]]
add_executable(TensorArray::Interpreter ALIAS tensorarray_interpreter)
