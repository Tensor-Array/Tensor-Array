cmake_minimum_required(VERSION 3.18)

file(GLOB TensorArray_src "*.cc" "*.c")
file(GLOB TensorArray_inc "*.hh" "*.h")

install(
    FILES ${TensorArray_inc}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensor-array/interp
    COMPONENT headers)

add_executable(tensorarray_interp SHARED ${TensorArray_src})

target_include_directories(tensorarray_interp PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tensorarray_interp TensorArray::Core)

set_property(TARGET tensorarray_interp PROPERTY C_STANDARD 11)
set_property(TARGET tensorarray_interp PROPERTY C_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_interp PROPERTY C_EXTENSIONS OFF)

set_property(TARGET tensorarray_interp PROPERTY CXX_STANDARD 17)
set_property(TARGET tensorarray_interp PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tensorarray_interp PROPERTY CXX_EXTENSIONS OFF)

install(
    TARGETS tensorarray_interp
    EXPORT TensorArrayTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array
    COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/tensor-array/interp
    COMPONENT Development)
