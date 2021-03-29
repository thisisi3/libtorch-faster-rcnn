cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp_frcnn)
set(CMAKE_CXX_STANDARD 14)

enable_language(CUDA)
add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
add_definitions(-DWITH_CUDA)


find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(cvops)
add_subdirectory(frcnn)

add_executable(train train.cpp)
target_link_libraries(train ${TORCH_LIBRARIES} frcnn cvops)

add_executable(test test.cpp)
target_link_libraries(test ${TORCH_LIBRARIES} frcnn cvops)

