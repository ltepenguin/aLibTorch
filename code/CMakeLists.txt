
add_library(aLibTorch SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torchtype.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
)

# headers
target_include_directories(aLibTorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# aOpenGL
target_include_directories(aLibTorch PUBLIC ${aOpenGL_INCLUDE_DIRS})
target_link_libraries(aLibTorch PUBLIC ${aOpenGL_LIBRARIES})

# Libtorch
target_include_directories(aLibTorch PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(aLibTorch PUBLIC ${TORCH_LIBRARIES})