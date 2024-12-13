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

set(BUILD_TESTING_OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xsimd Eigen nanobench)


add_library(spreadkernel src/spreadkernel.cpp src/polyfit.cpp)
target_include_directories(spreadkernel PUBLIC include)
target_link_libraries(spreadkernel PRIVATE xsimd Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(spreadkernel PRIVATE -march=native)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE spreadkernel nanobench Eigen3::Eigen)
target_compile_options(benchmark PRIVATE -march=native)

add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE spreadkernel)
