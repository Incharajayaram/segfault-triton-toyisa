cmake_minimum_required(VERSION 3.18)
project(tritonflow_emu_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE pybind11_DIR
    RESULT_VARIABLE pybind11_STATUS
)
if(pybind11_STATUS EQUAL 0 AND EXISTS "${pybind11_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()

find_package(pybind11 REQUIRED)

# The C++ emulator library (static, linked into the Python module).
add_library(emu_core STATIC
    machine.cpp
)
target_include_directories(emu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The Python extension module.
pybind11_add_module(_emu_cpp
    bindings.cpp
)
target_link_libraries(_emu_cpp PRIVATE emu_core)

# Install the extension into the emu/ package directory.
install(TARGETS _emu_cpp
    LIBRARY DESTINATION tritonflow/emu
)
