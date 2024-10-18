#include<onnxruntime_cxx_api.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <chrono>
#include <unordered_map>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;


#define pi acos(-1)
std::vector<std::string> labels1 = {"rice","barley","sunflower"};
std::vector<std::string> labels2 = {"beet","corn","cotton","pumpkin","sorghum","soybean","spinach","watermelon","wheat","cowpea"};
std::vector<std::string> labels3 = {"millet"};

// 全局變量，保存模型環境和會話
Ort::Env* env1 = nullptr;
Ort::Session* session1 = nullptr;
Ort::Env* env2 = nullptr;
Ort::Session* session2 = nullptr;
Ort::Env* env3 = nullptr;
Ort::Session* session3 = nullptr;

// Lazy initialization of session options
Ort::SessionOptions& get_session_options() {
    static Ort::SessionOptions session_options;
    static bool is_initialized = false;

    if (!is_initialized) {
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        is_initialized = true;
    }

    return session_options;
}

void initialize_model(const std::string& onnx_path_name1,const std::string& onnx_path_name2,const std::string& onnx_path_name3) {
    try {
        if (!env1) {
            env1 = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
        }

        Ort::SessionOptions& session_options1 = get_session_options();  // Use the lazy initialized session_options

        if (!session1) {
            //std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
            session1 = new Ort::Session(*env1, onnx_path_name1.c_str(), session_options1);
            std::cout << "Model session1 created successfully." << std::endl;
        }

        if (!env2) {
            env2 = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
        }

        Ort::SessionOptions& session_options2 = get_session_options();  // Use the lazy initialized session_options

        if (!session2) {
            //std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
            session2 = new Ort::Session(*env2, onnx_path_name2.c_str(), session_options2);
            std::cout << "Model session2 created successfully." << std::endl;
        }

         if (!env3) {
            env3 = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
        }

        Ort::SessionOptions& session_options3 = get_session_options();  // Use the lazy initialized session_options

        if (!session3) {
            //std::wstring modelPath = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
            session3 = new Ort::Session(*env3, onnx_path_name3.c_str(), session_options3);
            std::cout << "Model session3 created successfully." << std::endl;
        }

        


    } catch (const Ort::Exception& e) {
        std::cerr << "Error during ONNX Runtime initialization: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred during initialization." << std::endl;
    }
}

void FindNeibourAndDrawBox(std::vector<cv::Rect>& boxes,double distance_threshold,cv::Mat& image)
{
    int N = boxes.size();  // 矩形數量
    cv::Mat dist_matrix = cv::Mat::zeros(N, N, CV_64F);  // 創建距離矩陣，初始為0

    // 計算每兩個矩形之間的歐氏距離
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = boxes[i].x - boxes[j].x;
            double dy = boxes[i].y - boxes[j].y;
            double dist = std::sqrt(dx * dx + dy * dy);  // 計算距離
            dist_matrix.at<double>(i, j) = dist;
            dist_matrix.at<double>(j, i) = dist;  // 矩陣是對稱的
        }
    }
    // 創建一個掩碼矩陣，符合條件的元素設為 1，其他為 0
    cv::Mat mask = (dist_matrix < distance_threshold) & (dist_matrix > 0);  // 去掉對角線上的 0

    // 找到符合條件的坐標對
    std::vector<cv::Point> indices;
    cv::findNonZero(mask, indices);  // 獲取矩陣中非零元素的索引

    // 繪製符合條件的框
    for (const auto& idx : indices) {
        cv::rectangle(image, boxes[idx.x], cv::Scalar(0, 255, 0), 2);  // 綠色框
        cv::rectangle(image, boxes[idx.y], cv::Scalar(0, 255, 0), 2);  // 綠色框
    }

}


//define a struct to save some information
typedef struct {
	cv::RotatedRect box;
	float score;
	int Classindex;
}RotatedBOX;
//yolov8旋轉框目標檢測
std::pair<cv::Mat, int> run_inference1(std::vector<std::string>& labels, cv::Mat& frame, bool draw_point) {
    if (!session1) {
        std::cerr << "Model1 not initialized!" << std::endl;
        exit(1);
    }

    // 獲取模型輸入和輸出信息
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session1->GetInputCount();
    size_t numOutputNodes = session1->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 解析模型輸入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session1->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session1->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 解析模型輸出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session1->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];
    output_w = output_dims[2];
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session1->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    // 處理輸入圖像
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // 計算縮放因子
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);

    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    // 進行推理
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session1->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
    input_tensor_.release();
    image.release();

    // 解析輸出結果
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    cv::Mat det_output = dout.t();// 8400x5

    // post-process
	std::vector<cv::RotatedRect> boxes;
	std::vector<float> confidences;
    std::vector<RotatedBOX>BOXES;
    std::vector<int>class_list;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        RotatedBOX BOX;

		// confidence between 0 and 1
		if (score > 0.05)
		{
			float cx = det_output.at<float>(i, 0)*x_factor;
			float cy = det_output.at<float>(i, 1)* y_factor;
			float ow = det_output.at<float>(i, 2)*x_factor;
			float oh = det_output.at<float>(i, 3)* y_factor;
            float angle=det_output.at<float>(i,det_output.cols-1);
            float as_ratio=std::max(ow/oh,oh/ow);
            

            //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
            if (angle>=0.5*pi && angle <= 0.75*pi)
            {
                angle=angle-pi;
            }
            if (as_ratio<1.65)
            {
                continue;
            }
            BOX.Classindex=classIdPoint.x;
            class_list.push_back(classIdPoint.x);
            BOX.score=score; 
            cv::RotatedRect box=cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
            BOX.box=box;
            boxes.push_back(box);
            BOXES.push_back(BOX); 
			confidences.push_back(score);
		}
	}
    blob.release();
    ort_outputs.clear();

    // NMS accoding to each class
    std::vector<RotatedBOX> Remain_boxes;
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes,confidences,0.2,0.3, nms_result);
    if (nms_result.size()==0)
    {
        return {frame,0};
    }


	for (int i=0;i< nms_result.size();i++)
	{
		int index=nms_result[i];
		RotatedBOX Box_=BOXES[index];
		Remain_boxes.push_back(Box_);

	}
 




    // visulize the rotated bounding box result
    for (unsigned long i = 0; i < Remain_boxes.size(); ++i)
    {
       
        RotatedBOX Box_=Remain_boxes[i];
        cv::RotatedRect box_=Box_.box;
        cv::Point2f points[4];
        box_.points(points);
        int class_id=Box_.Classindex;

        if (draw_point)
        {
            float x1=points[0].x;
            float y1=points[0].y;
            float x2=points[1].x;
            float y2=points[1].y;
            float x3=points[2].x;
            float y3=points[2].y;
            float x4=points[3].x;
            float y4=points[3].y;
            float w=pow(pow(x1-x2,2)+pow(y1-y2,2),0.5);
            float h=pow(pow(x3-x2,2)+pow(y3-y2,2),0.5);
            float area=w*h;
            
            float cx,cy;
            cx=(x1+x2+x3+x4)/4;
            cy=(y1+y2+y3+y4)/4;
            int raius;
            if (area<600)
            {
                raius=5;
            }
            else if (area>1500)
            {
                raius=9;
            }
            else
            {
                raius=7;
            }
            cv::circle(frame,cv::Point2f(cx,cy),raius,cv::Scalar(0, 0, 255),-1);

        }
        else
        {
            for (int i = 0; i < 4; ++i) 
            {
                cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);  
            }
            cv::putText(frame, labels1[class_id], points[0], cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 1);
        }

        
    }

    boxes.clear();
    confidences.clear();
    Remain_boxes.clear();
    

    return {frame, nms_result.size()};
}



// 推理函數，重複使用已經初始化的模型
std::pair<cv::Mat, int> run_inference2(std::vector<std::string>& labels, cv::Mat& frame, bool draw_point) {
    if (!session2) {
        std::cerr << "Model2 not initialized!" << std::endl;
        exit(1);
    }

    // 獲取模型輸入和輸出信息
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session2->GetInputCount();
    size_t numOutputNodes = session2->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 解析模型輸入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session2->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session2->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 解析模型輸出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session2->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];
    output_w = output_dims[2];
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session2->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    // 處理輸入圖像
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // 計算縮放因子
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);

    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    // 進行推理
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session2->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
    input_tensor_.release();
    image.release();
    // 解析輸出結果
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    cv::Mat det_output = dout.t();

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, 4 + labels.size());
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        if (score > 0.05) {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);

            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }
    
    blob.release();
    ort_outputs.clear();

    if(boxes.size()==0)
    {
        return {frame, 0};
    }

    // NMS（非極大值抑制）
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.05, 0.3, indexes);

    std::vector<cv::Rect> newboxes;
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        newboxes.push_back(boxes[index]);
        if (draw_point) {
            cv::Rect box = boxes[index];
            float cx, cy, area;
            area = box.width * box.height;
            cx = box.x + box.width / 2;
            cy = box.y + box.height / 2;
            int raius;
            if (area < 600) {
                raius = 4;
            } else if (area > 1500) {
                raius = 8;
            } else {
                raius = 6;
            }
            cv::circle(frame, cv::Point2f(cx, cy), raius, cv::Scalar(0, 0, 255), -1);
        } else {
            cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
            putText(frame, labels[classIds[index]], cv::Point(boxes[index].tl().x, boxes[index].tl().y),
                cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        }
    }
    
    FindNeibourAndDrawBox(newboxes,15,frame);
    boxes.clear();
    classIds.clear();
    confidences.clear();
    newboxes.clear();

    

    return {frame, indexes.size()};
}


std::pair<cv::Mat, int> run_inference3(std::vector<std::string>& labels, cv::Mat& frame, bool draw_point) {
    if (!session3) {
        std::cerr << "Model3 not initialized!" << std::endl;
        exit(1);
    }

    // 獲取模型輸入和輸出信息
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session3->GetInputCount();
    size_t numOutputNodes = session3->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 解析模型輸入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session3->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session3->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 解析模型輸出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session3->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];
    output_w = output_dims[2];
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session3->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    // 處理輸入圖像
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // 計算縮放因子
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);

    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    // 進行推理
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session3->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    input_tensor_.release();
    image.release();
    // 解析輸出結果
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    cv::Mat det_output = dout.t();

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, 4 + labels.size());
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        if (score > 0.05) {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);

            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    blob.release();
    ort_outputs.clear();
    if(boxes.size()==0)
    {
        return {frame, 0};
    }

    // NMS（非極大值抑制）
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.05, 0.3, indexes);

    std::vector<cv::Rect> newboxes;
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        newboxes.push_back(boxes[index]);
        if (draw_point) {
            cv::Rect box = boxes[index];
            float cx, cy, area;
            area = box.width * box.height;
            cx = box.x + box.width / 2;
            cy = box.y + box.height / 2;
            int raius;
            if (area < 600) {
                raius = 4;
            } else if (area > 1500) {
                raius = 8;
            } else {
                raius = 6;
            }
            cv::circle(frame, cv::Point2f(cx, cy), raius, cv::Scalar(0, 0, 255), -1);
        } else {
            cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
            putText(frame, labels[classIds[index]], cv::Point(boxes[index].tl().x, boxes[index].tl().y),
                cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        }
    }
    
    FindNeibourAndDrawBox(newboxes,15,frame);

    boxes.clear();
    classIds.clear();
    confidences.clear();
    newboxes.clear();


    return {frame, indexes.size()};
}

// 程序結束時釋放模型
void release_model() {
    if (session1) {
        delete session1;
        session1 = nullptr;
    }
    if (env1) {
        delete env1;
        env1 = nullptr;
    }

    if (session2) {
        delete session2;
        session2 = nullptr;
    }
    if (env3) {
        delete env3;
        env3 = nullptr;
    }
}

std::pair<cv::Mat, int> detect(std::string &img_name,  int &task,bool& vis,std::string img_savepath="scratch_file", std::string predict_savepath="scratch_file")
{
	cv::Mat input_image = cv::imread(img_name);
   
    assert(!input_image.empty());
	int ih = input_image.rows;
	int iw = input_image.cols;
	int64 start = cv::getTickCount();
    
     // 初始化模型（只加載一次）
    std::string model_path1 = "/home/kingargroo/seed/ablation1/rotate.onnx";
    std::string model_path2 = "/home/kingargroo/seed/ablation1/normal.onnx";
    std::string model_path3 = "/home/kingargroo/seed/ablation1/small.onnx";
    initialize_model(model_path1,model_path2,model_path3);
	
	try {
		if (task==0)
		{
            auto [output_image,count_number] = run_inference1(labels1, input_image, true);
            if (vis)
            {
                cv::imwrite("test.jpg",output_image);
            }
			return {output_image, count_number};
		}
		else if (task==1)
		{
            auto [output_image, count_number] = run_inference2(labels2, input_image, true);
            if (vis)
            {
                cv::imwrite("test.jpg",output_image);
            }
			return {output_image, count_number};
            
		}
		else if (task==2)
		{
            
			auto [output_image, count_number] = run_inference3(labels3, input_image, true);
            if (vis)
            {
                cv::imwrite("test.jpg",output_image);
            }
			return {output_image, count_number};

		}
		else
		{
			throw task;
		}
	}
	catch(int)
	{ 
		std::cout<<"task value error, task value must be one of 0,1 or 2"<<std::endl;
		exit(1);
	}
     // 釋放模型
    release_model();
    input_image.release();
	
}


int main() {
   
    //std::string dir_path="/home/kingargroo/seed/pic";
    bool vis=true;
    int task=1;
    // for (const auto & entry:fs::directory_iterator(dir_path))
    // {
    //     std::string img_name=entry.path().filename().string();
    //     std::string img_name_=dir_path+"/"+img_name;
    //     std::cout<<img_name_<<std::endl;
    //     auto [output_image, count_number]=detect(img_name_,  task,vis);
    //     std::cout<< count_number<<std::endl;

    // }
    std::string img_name_="/home/kingargroo/seed/validate/348.jpg";
    auto [output_image, count_number]=detect(img_name_,  task,vis);
    std::cout<< count_number<<std::endl;
    return 0;
}
