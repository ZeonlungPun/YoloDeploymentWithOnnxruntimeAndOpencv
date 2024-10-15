# 介紹

本倉庫包含了一些流行的物體目標檢測和其他計算機視覺任務的 C++ 部署實現。

C++在部署模型進行應用上具有舉足輕重的地位。

我們現在主要在 Linux 平台上提供了基於 OpenCV 4.8、onnxruntime 和 libtorch 框架的推理代碼，並附有 CMakeList 文件。

# 新聞
[2024/10/1]：我們發布了一個綜合目標檢測（如Yolov5、Yolov8和Yolov11）推理的實現，這些模型由Ultralytics開發，使用onnxruntime。代碼可在此處找到。 
</br> 
[2024/5/6]：我們發布了一個YoloT的實現，用於遙感領域中的目標檢測。代碼可在此處找到。
</br> 
[2024/3/6]：我們發布了一個新的Yolov8-obb實現，用於使用onnxruntime進行定向邊界框目標檢測。代碼可在此處找到。 
</br> 
[2024/3/2]：我們發布了一個新的Yolov8-obb實現，使用opencv4.8進行定向邊界框目標檢測。代碼可在此處找到。 
</br> 
[2023/3/2]：我們發布了一個Yolov8推理的實現，僅使用libtorch。代碼可在此處找到。

# 功能

目前的模型包括用於物體檢測的 Yolov5、Yolov7、Yolov8、RTDETR。

用於衛星影像中小物體檢測的 YoloT。

用於旋轉物體檢測的 Yolov8obb。

用於自動駕駛的 YoloP。

用於關鍵點檢測的 Yolov8pose。

用於實例分割的 YOLOV8-instance-seg 和 yolov5-seg.cpp。

# 外部庫版本
onnxruntime-linux-x64-1.16.3

opencv 4.8

libtorch 2.1.2

# CUDA
在 onnxruntime 中使用 CUDA：
```
OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
```

在 OpenCV 4.8 中使用 CUDA：
```
net = cv::dnn::readNetFromONNX(modelPath);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

# 平台差異
Linux 和 Windows 平台之間的差異（僅在讀取模型路徑時不同）

Linux：
```
Ort::Session session(env, onnx_path_name.c_str(), session_options);
```

Windows：
```
std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
Ort::Session session_(env, modelPath.c_str(), session_options);
```

# 應用舉例
1, 在 **cmakelist** 文件中, 你需要設定外部庫的安裝路徑，比如:
```
set(ONNXRUNTIME_DIR "/home/kingargroo/cpp/onnxruntimeYOLOV8OBB/onnxruntime-linux-x64-1.16.3")
include_directories("${ONNXRUNTIME_DIR}/include")
```
or:
```
find_package(OpenCV 4 REQUIRED)
```

2, 你需要將訓練好的 **.pt** 權重轉換爲 **onnx** 格式 (onnxruntime and opencv) 或者 **torchscript**(libtorch)  格式.
```
from ultralytics import YOLO
#Load a model
model = YOLO('yolov8n-obb.pt') # load an official model
#Export the model
model.export(format='onnx', imgsz=640, opset=12)
#or
model.export(format='torchscript', imgsz=640, opset=12)
```

3, 在 **.cpp** 文件中,你需要指定你的模型路徑、圖片路徑和類別名稱. 我們推薦你使用 **yoloUltralytics.cpp**,這是一個針對Ultralytics寫的C++推斷實現，比如:
```
int main() {
    // Load the labels and test the class with an image
    std::vector<std::string> labels = {"beet", "corn", "cotton", "pumpkin", "sorghum", "soybean", "spinach", "watermelon", "wheat", "cowpea"};
    std::string img_name = "/home/kingargroo/seed/validate/QYTC20240423666_20240921092551073.jpg";
    std::string onnx_path_name = "/home/kingargroo/seed/ablation1/normal.onnx";
    std::vector<std::string> preprocess_method = std::vector<std::string>{"direct"};
    // Load the image
    cv::Mat input_image = cv::imread(img_name);

    // Instantiate the inference class
    YolovUltralyticsInference inference(labels, onnx_path_name, input_image,preprocess_method);

    // Perform inference
    cv::Mat output_image = inference.main_process();

    // Save the output image
    if (!output_image.empty()) {
        cv::imwrite("/home/kingargroo/cpp/yolov8forseed/result.jpg", output_image);
        std::cout << "Inference completed and result saved." << std::endl;
    } else {
        std::cerr << "Inference failed." << std::endl;
    }

    return 0;
}
```



# 參考論文

Yolov7: https://arxiv.org/abs/2207.02696

RTDETR: https://arxiv.org/abs/2304.08069

YoloP: https://arxiv.org/abs/2108.11250

YOLACT/Yolov8seg: https://arxiv.org/abs/1904.02689

YoloT: https://arxiv.org/pdf/1805.09512.pdf

Yolopose/Yolov8pose: https://arxiv.org/abs/2204.06806

# 引用
如果您覺得我們的工作對你有所幫助，請給我們star或者在論文中引用我們的github：
```
@misc{YOLOVDeploymentC++,
  author       = {Zeonlung Pun},
  title        = {YOLOVc++},
  year         = {2024},
  howpublished = {\url{https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv}}
}
```

# 其它語言
[English Version](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/README.md) 

