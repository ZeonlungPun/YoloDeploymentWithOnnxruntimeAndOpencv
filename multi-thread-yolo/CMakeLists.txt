cmake_minimum_required(VERSION 3.5)

Project(yolov7)

find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH PATHS /home/punzeonlung/cvDemo/opencv480-linux/lib/cmake/opencv4)
find_package(Threads REQUIRED)
set(ONNXRUNTIME_DIR "/home/punzeonlung/CPP/onnxLinuxYOLOV7/onnxruntime-linux-x64-1.16.3")

include_directories("${ONNXRUNTIME_DIR}/include")

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(${OpenCV_INCLUDE_DIRS} ) 


add_executable(result yoloUltralyticsMultiThred.cpp )
target_link_libraries(result "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} Threads::Threads)

