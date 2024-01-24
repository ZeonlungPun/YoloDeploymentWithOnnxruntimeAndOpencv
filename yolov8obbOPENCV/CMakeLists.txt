cmake_minimum_required(VERSION 2.8)

Project(yolov8)

find_package(OpenCV 4 REQUIRED)

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(${OpenCV_INCLUDE_DIRS} ) 


add_executable(result yolov8obb.cpp )

target_link_libraries(result ${OpenCV_LIBS})