# 介紹

本倉庫包含了一些流行的物體目標檢測和其他計算機視覺任務的 C++ 部署實現。

我們現在主要在 Linux 平台上提供了基於 OpenCV 4.8、onnxruntime 和 libtorch 框架的推理代碼，並附有 CMakeList 文件。

# 功能

目前的模型包括用於物體檢測的 Yolov5、Yolov7、Yolov8、RTDETR。

用於衛星影像中小物體檢測的 YoloT。

用於旋轉物體檢測的 Yolov8obb。

用於自動駕駛的 YoloP。

用於關鍵點檢測的 Yolov8pose。

用於實例分割的 YOLOV8-instance-seg 和 yolov5-seg.cpp。

# 版本
onnxruntime-linux-x64-1.16.3

opencv 4.8

# CUDA
在 onnxruntime 中使用 CUDA：
OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

在 OpenCV 4.8 中使用 CUDA：
net = cv::dnn::readNetFromONNX(modelPath);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

#差異
Linux 和 Windows 平台之間的差異（僅在讀取模型路徑時不同）

Linux：
Ort::Session session(env, onnx_path_name.c_str(), session_options);

Windows：
std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
Ort::Session session_(env, modelPath.c_str(), session_options);

# 參考論文

Yolov7: https://arxiv.org/abs/2207.02696

RTDETR: https://arxiv.org/abs/2304.08069

YoloP: https://arxiv.org/abs/2108.11250

YOLACT/Yolov8seg: https://arxiv.org/abs/1904.02689

YoloT: https://arxiv.org/pdf/1805.09512.pdf

Yolopose/Yolov8pose: https://arxiv.org/abs/2204.06806

[English Version](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/README.md) 

