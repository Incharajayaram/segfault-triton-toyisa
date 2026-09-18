cmake_minimum_required(VERSION 3.18)
project(tritonflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/tritonflow/emu/cpp)
