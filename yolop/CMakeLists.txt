cmake_minimum_required(VERSION 2.8)

Project(yolovP)

find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH PATHS /home/punzeonlung/cvDemo/opencv480-linux/lib/cmake/opencv4)

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(${OpenCV_INCLUDE_DIRS} include) 


add_executable(result yolop.cpp)

target_link_libraries(result ${OpenCV_LIBS})
