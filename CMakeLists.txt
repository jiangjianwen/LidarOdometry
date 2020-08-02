cmake_minimum_required(VERSION 2.8.3)
project(LO)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")



find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(LO lidarOdometry.cpp)
target_link_libraries(LO ${catkin_LIBRARIES} ${PCL_LIBRARIES}  ${CERES_LIBRARIES} ${OpenCV_LIBS})



