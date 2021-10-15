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

# tensorboard logger
#find_package(Protobuf REQUIRED)
target_include_directories(aLibTorch PUBLIC 
    ${EXT_DIR}/tensorboard_logger/include
    ${EXT_DIR}/tensorboard_logger/build)

target_link_libraries(aLibTorch PUBLIC ${EXT_DIR}/tensorboard_logger/build/libtensorboard_logger.a)
target_link_libraries(aLibTorch PUBLIC ${Protobuf_LIBRARIES})

# Libtorch
target_include_directories(aLibTorch PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(aLibTorch PUBLIC ${TORCH_LIBRARIES})

# export
export(
    TARGETS aLibTorch
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_build/alt_config.cmake)