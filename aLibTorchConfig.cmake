# set external directory
set(EXT_DIR ${CMAKE_CURRENT_LIST_DIR}/ext)

# libtorch
set(Torch_DIR ${EXT_DIR}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/cmake_build/alt_config.cmake)