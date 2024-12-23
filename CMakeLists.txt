cmake_minimum_required(VERSION 3.17)
project(09-mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Download Eigen
include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/**/*.cpp" "${CMAKE_SOURCE_DIR}/src/**/*.h")

add_executable(09-mc ${SRC_FILES})

# Link package pthread
find_package(Threads REQUIRED)
target_link_libraries(09-mc PRIVATE Threads::Threads)

target_include_directories(09-mc PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_options(09-mc PRIVATE -Ofast -march=native)
target_link_libraries(09-mc PRIVATE Eigen3::Eigen)
