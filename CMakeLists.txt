cmake_minimum_required(VERSION 3.10)

# set project
project(aLibTorch VERSION 0.1.0)
set(CMAKE_CXX_COMPLIER "g++")
set(CMAKE_CXX_STANDARD 17)

# set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set external directory
set(EXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)

# libtorch
set(Torch_DIR ${EXT_DIR}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# aOpenGL
set(aOpenGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../aOpenGL)
find_package(aOpenGL REQUIRED)

# multithreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# source codes
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)