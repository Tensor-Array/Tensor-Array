block(SCOPE_FOR POLICIES)
    enable_language(C)
    
    file(
        GLOB TensorArray_Interpreter_src
        "${PROJECT_SOURCE_DIR}/src/tensor-array/interp/*.c"
        "${PROJECT_SOURCE_DIR}/src/tensor-array/interp/*.cc"
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
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
        COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/interp
        COMPONENT Development)
    #[[
    add_custom_command(
        OUTPUT test.tmp
        DEPENDS tensorarray_interpreter
        POST_BUILD
        COMMAND tensorarray_interpreter)
    ]]
    add_executable(TensorArray::Interpreter ALIAS tensorarray_interpreter)

endblock()
