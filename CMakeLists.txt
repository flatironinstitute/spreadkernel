cmake_minimum_required(VERSION 3.14)
project(
  spreadkernel
  LANGUAGES CXX
  )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG v4.3.11
  GIT_SHALLOW TRUE)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 13.0.0
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(xsimd nanobench)

add_library(spreadkernel src/spreadkernel.cpp src/polyfit.cpp)
target_include_directories(spreadkernel PUBLIC include)
target_link_libraries(spreadkernel PRIVATE xsimd OpenMP::OpenMP_CXX)
target_compile_options(spreadkernel PRIVATE -march=native)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE spreadkernel nanobench)
target_compile_options(benchmark PRIVATE -march=native)

add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE spreadkernel)
