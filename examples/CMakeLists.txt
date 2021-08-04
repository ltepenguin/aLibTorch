# example 01
add_executable(mnist ${CMAKE_CURRENT_SOURCE_DIR}/01_mnist.cpp)
target_link_libraries(mnist PUBLIC aLibTorch aOpenGL)

# example 02
# add_executable(dataloader ${CMAKE_CURRENT_SOURCE_DIR}/02_dataloader.cpp)
# target_link_libraries(dataloader PUBLIC aLibTorch)

# example 03
add_executable(python ${CMAKE_CURRENT_SOURCE_DIR}/03_python.cpp)
target_link_libraries(python PUBLIC aLibTorch aOpenGL)

# example 04
add_executable(footcontact ${CMAKE_CURRENT_SOURCE_DIR}/04_footcontact.cpp)
target_link_libraries(footcontact PUBLIC aLibTorch aOpenGL)
