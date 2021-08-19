add_library(aLibTorch SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pymodule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torchtype.cpp
)

# headers
target_include_directories(aLibTorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# aOpenGL
target_link_libraries(aLibTorch PRIVATE aOpenGL)

# Libtorch
target_include_directories(aLibTorch PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(aLibTorch PUBLIC ${TORCH_LIBRARIES})

# export
export(
    TARGETS aLibTorch
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_build/alt_config.cmake)