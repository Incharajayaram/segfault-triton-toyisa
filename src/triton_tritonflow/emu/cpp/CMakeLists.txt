cmake_minimum_required(VERSION 3.18)
project(tritonflow_emu_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11.
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
    LIBRARY DESTINATION triton_tritonflow/emu
)
