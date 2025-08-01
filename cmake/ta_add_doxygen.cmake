find_package(Doxygen)

if(Doxygen_FOUND)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_FILE_PATTERNS *.c *.cc *.h *.hh *.cu *.md)
    set(DOXYGEN_EXTENSION_MAPPING "*.cu=c++")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
    set(DOXYGEN_EXCLUDE_PATTERNS ".*/*" "build/*" "cmake/*" "scripts/*" "CMakeLists.txt" "*.cmake" "*.cmake.*")

    doxygen_add_docs(tensorarray_docs ${PROJECT_SOURCE_DIR} ALL)
endif()
