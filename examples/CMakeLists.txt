# example 01
add_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/01_mnist.cpp)
target_link_libraries(mnist PUBLIC aLibTorch)
