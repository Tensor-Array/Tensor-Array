find_package(Doxygen)

if(Doxygen_FOUND)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_FILE_PATTERNS *.c *.cc *.h *.hh *.cu)
    set(DOXYGEN_EXTENSION_MAPPING "*.cu=c++")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")

    doxyen_add_docs(tensorarray_docs ${PROJECT_SOURCE_DIR}/src ALL)
endif()
