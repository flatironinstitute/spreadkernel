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

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
  GIT_SHALLOW TRUE)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0
  GIT_SHALLOW TRUE)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
  set_target_properties(spdlog
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

FetchContent_MakeAvailable(xsimd nanobench doctest spdlog)

add_library(spreadkernel SHARED src/spreadkernel.cpp src/polyfit.cpp src/doctest.cpp)
target_include_directories(spreadkernel PUBLIC include)
target_link_libraries(spreadkernel PRIVATE xsimd OpenMP::OpenMP_CXX doctest spdlog::spdlog)
target_compile_options(spreadkernel PRIVATE -march=native)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE spreadkernel nanobench)
target_compile_options(benchmark PRIVATE -march=native)

add_executable(run_tests test/run_tests.cpp)
target_link_libraries(run_tests PRIVATE spreadkernel doctest)
