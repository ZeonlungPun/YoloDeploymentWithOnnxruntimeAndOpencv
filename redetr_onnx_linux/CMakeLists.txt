cmake_minimum_required(VERSION 2.8)

Project(yolov8obb)

find_package(OpenCV 4 REQUIRED)

set(ONNXRUNTIME_DIR "/home/kingargroo/cpp/onnxruntimeYOLOV8OBB/onnxruntime-linux-x64-1.16.3")

include_directories("${ONNXRUNTIME_DIR}/include")

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(${OpenCV_INCLUDE_DIRS} ) 


add_executable(result rtdetr.cpp )
target_link_libraries(result "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} )
target_link_libraries(result ${OpenCV_LIBS})



