# Introduction

This repository contains the **C++** inference  implentation of some popular object detection and other known computer vision tasks.

C++ is a key language for deployment.

Now we offer the implentation of **Opencv 4.8**, **onnxruntime** and **libtorch** framework inference codes mainly on **linux** platform with cmakelist files in the **CPU** environment.

# News
[2024/10/1]: We release a implentation for the comprehensive **object detection** (such as Yolov5 , Yolov8 and  Yolov11) inference developed by [Ultralytics](https://github.com/ultralytics/ultralytics) using **onnxruntime**. Code is available [here](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/yoloUltralytics.cpp).
</br>
[2024/3/6]: We release a implentation for the new release **YOLOv8-obb** for oriented bounding box object detection using **onnxruntime**. Code is available [here](https://github.com/IDEA-Research/OpenSeeD](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/onnxruntimeYOLOV8OBB).
</br>
[2024/3/2]: We release a implentation for the new release **YOLOv8-obb** for oriented bounding box object detection using only **opencv4.8**. Code is available [here](https://github.com/IDEA-Research/OpenSeeD](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/yolov8obbOPENCV).


# Function

The models now include Yolov5 , Yolov7, Yolov8 ,RTDETR for object detection.

YoloT for small object detection In Satellite Imagery.

Yolov8obb for rotated object detection.

YoloP for Autopilot。

Yolov8pose for key points detection.

YOLOV8-instance-seg and yolov5-seg.cpp for instance segemention.

# Version


onnxruntime-linux-x64-1.16.3

opencv 4.8

# CUDA

Using CUDA with onnxruntime:
```
OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
```
Using CUDA with opencv 4.8:
```
net = cv::dnn::readNetFromONNX(modelPath);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

# Difference
The difference between Linux and Windows Platform (just different when reading model path)

Linux :
```
 Ort::Session session(env, onnx_path_name.c_str(), session_options);
```

Windows:
```
std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
Ort::Session session_(env, modelPath.c_str(), session_options);
```
# Reference Paper

Yolov7: https://arxiv.org/abs/2207.02696

RTDETR: https://arxiv.org/abs/2304.08069

YoloP : https://arxiv.org/abs/2108.11250

YOLACT/Yolov8seg:https://arxiv.org/abs/1904.02689

YoloT: https://arxiv.org/pdf/1805.09512.pdf

Yolopose/yolov8pose: https://arxiv.org/abs/2204.06806

# Example

# Citation

[中文版](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/README_ch.md) 
