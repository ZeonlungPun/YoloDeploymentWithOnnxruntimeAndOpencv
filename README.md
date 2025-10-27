# 🚀 Introduction

This repository contains the **C++** inference  implentation of some popular object detection and other known computer vision tasks.

C++ is a key language for deployment.

Now we offer the implentation of **Opencv 4.8**, **onnxruntime** and **libtorch** framework inference codes mainly on **linux** platform with cmakelist files in the **CPU** environment.

# ✨ Project News and Updates
| Date           | Content                                                                                  | Code Link                                 |
|----------------|------------------------------------------------------------------------------------------|-------------------------------------------|
| [2025/10/1]    | Released the **ONNX Runtime** implementation for **Yolo-World**—a Yolov8-based open-vocabulary object detection model. | [Yolo-World Code](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/yolo-world) |
| [2024/10/1]    | Released the **ONNX Runtime** implementation for comprehensive **object detection** models (Yolov5, Yolov8, Yolov11) developed by [Ultralytics](https://github.com/ultralytics/ultralytics). | [YoloUltralytics.cpp](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/yoloUltralytics.cpp) |
| [2024/5/6]     | Released the implementation for **YoloT**, used for **remote sensing** object detection. | [YoloT.cpp](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/YoloT.cpp) |
| [2024/3/6]     | Released the **ONNX Runtime** implementation for the newly released **Yolov8-obb** (oriented bounding box object detection). | [YOLOV8OBB - ONNX Runtime](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/onnxruntimeYOLOV8OBB) |
| [2024/3/2]     | Released the **OpenCV 4.8** implementation for **Yolov8-obb**. | [YOLOV8OBB - OpenCV](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/yolov8obbOPENCV) |
| [2023/3/2]     | Released the **LibTorch-only** inference implementation for **Yolov8**. | [Yolov8 LibTorch](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/tree/main/yolov8lib-torch) |



# Function

The models now include Yolov5 , Yolov7, Yolov8 ,RTDETR for object detection.

YoloT for small object detection In Satellite Imagery.

Yolov8obb for rotated object detection.

Yolo-World for open-vocabulary object detection.

YoloP for Autopilot.

Yolov8pose for key points detection.

YOLOV8-instance-seg and yolov5-seg.cpp for instance segemention.

# Package Version


onnxruntime-linux-x64-1.16.3

opencv 4.8

libtorch 2.1.2

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

# Example
1, In the **cmakelist** files, you should change the path of specific package we need to include, such as:
```
set(ONNXRUNTIME_DIR "/home/kingargroo/cpp/onnxruntimeYOLOV8OBB/onnxruntime-linux-x64-1.16.3")
include_directories("${ONNXRUNTIME_DIR}/include")
```
or:
```
find_package(OpenCV 4 REQUIRED)
```

2, You need to transform the trained weight or checkpoint to the **onnx** format (onnxruntime and opencv) or **torchscript**(libtorch) format.
```
from ultralytics import YOLO
#Load a model
model = YOLO('yolov8n-obb.pt') # load an official model
#Export the model
model.export(format='onnx', imgsz=640, opset=12)
#or
model.export(format='torchscript', imgsz=640, opset=12)
```

3, In the **.cpp** file, you need to write your own model path, image path and label names. We recommend you to use the **yoloUltralytics.cpp**, which is a general implementation for object detection. For example:
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
# Related Blogs

My experience about onnx deployment: https://medium.com/@zeonlungpun/how-to-avoid-unlimited-memory-accumulation-when-using-onnxruntime-c-for-yolov8-inference-6b8dba60058c

introduction of libtroch,onnxruntime and opecv: https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-1-some-hit-frameworks-and-their-usages-e31bf6e60a30

deployment with object detection:  https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-2-object-detection-model-22b877b79737

deployment with semantic segmentation: https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-3-semantic-segmentation-model-883fd557126f

deployment with instance segmentation: https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-4-instance-segmentation-model-f6ef3d8a7725

deployment with oriented object detection: https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-5-oriented-object-detection-c5dc1210dd25

deployment with key points detection (Pose estimation):https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-6-key-points-detection-pose-estimation-f28a057bfe1b

deployment with open-vocabulary object detection (Yolo-World): https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-7-open-vocabulary-object-detection-49e29397bc81

# Reference Paper

Yolov7: https://arxiv.org/abs/2207.02696

RTDETR: https://arxiv.org/abs/2304.08069

YoloP : https://arxiv.org/abs/2108.11250

YOLACT/Yolov8seg:https://arxiv.org/abs/1904.02689

YoloT: https://arxiv.org/pdf/1805.09512.pdf

Yolopose/yolov8pose: https://arxiv.org/abs/2204.06806



# Citation
If you think our wokr is helpful, please give us a star or citate our repository:
```
@misc{YOLOVDeploymentC++,
  author       = {Zeonlung Pun},
  title        = {YOLOVc++},
  year         = {2024},
  howpublished = {\url{https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv}}
}
```

# Other Language

[中文版](https://github.com/ZeonlungPun/YoloDeploymentWithOnnxruntimeAndOpencv/blob/main/README_ch.md) 
