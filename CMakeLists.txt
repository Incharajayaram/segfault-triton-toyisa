cmake_minimum_required(VERSION 3.18)
project(triton_toyisa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/triton_toyisa/emu/cpp)
