cmake_minimum_required(VERSION 3.20)
project(qisb_engine LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(COMMON_FLAGS -O3 -DNDEBUG -DEIGEN_NO_DEBUG -march=native -funroll-loops)

add_executable(backtest_engine src/main_backtest.cpp)
target_include_directories(backtest_engine PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(backtest_engine PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(backtest_engine PRIVATE ${COMMON_FLAGS})

add_executable(live_engine src/main_live.cpp)
target_include_directories(live_engine PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(live_engine PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX Threads::Threads rt)
target_compile_options(live_engine PRIVATE ${COMMON_FLAGS})
