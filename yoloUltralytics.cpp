/*
 * YolovUltralyticsInference: A YOLOv8-based inference class for object detection
 * 
 * Author: Zeonlung Pun
 * Date: October 2, 2024
 * 
 * Description:
 * This class provides an implementation of Ultralytics YOLO series object detection algorithms inference 
 * using ONNX and OpenCV.
 * It includes methods for reading the model, preprocessing input images, running inference,
 * and drawing detection results on the input image.
 * 
 * Usage:
 * The YolovUltralyticsInference class is instantiated with the ONNX model path,
 * input image, and optional preprocessing and threshold parameters.
 * The `main_process` function performs the complete inference and returns the processed image.
 *
 * Note:
 * This code assumes the ONNX Runtime and OpenCV libraries are properly installed and configured.
 * 
 * Contact: zeonlungpun@gmail.com
 * 
 */


#include<onnxruntime_cxx_api.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>


class YolovUltralyticsInference {
    /*
	Initializes an instance of the YOLOv8 class.
	Args:
		onnx_path_name: Path to the ONNX model.
		input_image: the original input image.
		confidence_thres: Confidence threshold for filtering detections.
		iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
		labels: Load your own model's class names
	*/
public:
    std::vector<std::string> labels;
    std::string onnx_path_name;
    cv::Mat input_image, result_image;
    float confidence_thres;
    float iou_thres;
    int model_input_w, model_input_h, model_output_h, model_output_w;
    float x_factor, y_factor;
    std::vector<std::string> preprocess_method;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

    YolovUltralyticsInference(
        std::vector<std::string> labels,
        std::string onnx_path_name,
        cv::Mat input_image,
        std::vector<std::string> preprocess_method = std::vector<std::string>{"direct"},
        float confidence_thres = 0.05,
        float iou_thres = 0.3)
        : labels(labels), onnx_path_name(onnx_path_name), input_image(input_image),
          preprocess_method(preprocess_method), confidence_thres(confidence_thres),
          iou_thres(iou_thres), env(ORT_LOGGING_LEVEL_ERROR, "yolov-onnx"), session_options(),
          session(nullptr)
    {
        this->model_input_w = 0;
        this->model_input_h = 0;
        this->model_output_h = 0;
        this->model_output_w = 0;
        this->x_factor = 0;
        this->y_factor = 0;

        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        read_model();
    }

    void read_model() {
        std::ifstream infile(onnx_path_name);
        if (!infile.good()) {
            throw std::runtime_error("ONNX model file not found: " + onnx_path_name);
        }

        session = Ort::Session(env, onnx_path_name.c_str(), session_options);

        size_t numInputNodes = session.GetInputCount();
        size_t numOutputNodes = session.GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input information
        for (size_t i = 0; i < numInputNodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();

            this->model_input_w = input_dims[3];
            this->model_input_h = input_dims[2];
            std::cout << "Input format: BatchxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x"
                      << input_dims[2] << "x" << input_dims[3] << std::endl;
        }

        // Get output information
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        this->model_output_h = output_dims[1];
        this->model_output_w = output_dims[2];
        std::cout << "Output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;

        for (size_t i = 0; i < numOutputNodes; i++) {
            auto out_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(out_name.get());
        }

        this->x_factor = this->input_image.cols / static_cast<float>(this->model_input_w);
        this->y_factor = this->input_image.rows / static_cast<float>(this->model_input_h);
    }

    cv::Mat preprocess() {
               /*
	Preprocesses the input image before performing inference.
	prepocess_method:the image preprocess method,including direct,letter_box, copy_paste
	
	Returns:
		None
	*/
        this->result_image = this->input_image.clone();
        cv::Mat blob;

        // Convert the image color space from BGR to RGB
        cv::cvtColor(this->input_image, this->input_image, cv::COLOR_BGR2RGB);

        // Preprocesses the input image according to different selected method
		if (this->preprocess_method== std::vector<std::string>{"direct"})
		{	
			//Resize the image to match the input shape
    		cv::resize(this->input_image,this->input_image,cv::Size(this->model_input_w,this->model_input_h));
			//normalize the image from [0,255] to [0,1]
			cv::dnn::blobFromImage(this->input_image,blob, 1 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);
		}
		else if (this->preprocess_method==std::vector<std::string>{"letter_box"} )
		{
			blob=0;
		}
		else if (this->preprocess_method==std::vector<std::string>{"copy_paste"} )
		{
			// width and height for raw input image 
			int raw_image_w = this->input_image.cols;
			int raw_image_h = this->input_image.rows;
			int _max = std::max(raw_image_h,raw_image_w);
			// temporary blanket image
			cv::Mat temp_image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
			//keep same w/h ratio
			cv::Rect roi(0, 0, raw_image_w, raw_image_h);
			this->input_image.copyTo(temp_image(roi));
			//1,normalize the image from [0,255] to [0,1]
			//2,resize the image to the specific size
			cv::dnn::blobFromImage(temp_image,blob, 1 / 255.0, cv::Size(this->model_input_w, this->model_input_h), cv::Scalar(0, 0, 0), true, false);
			
		}
		else{
			std::cout << "prepocess_method: ";
			for (const auto& method : this->preprocess_method) {
				std::cout << method << " ";}
			std::cout << "does not exist" << std::endl;
		}
		return blob;
    }

    cv::Mat draw_detections(cv::Mat img, std::vector<int> indexes, std::vector<cv::Rect> boxes, std::vector<int> classIds) {
        /*
	Draws bounding boxes and labels on the input image based on the detected objects.
	Args:
		img: The input image to draw detections on.
		boxes: Detected bounding box before NMS.
		classIds: Class ID for the detected object.
		indexes: sample ID after NMS.
	Returns:
		The image with some detection boxes on.
	*/
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int idx = classIds[index];
            cv::rectangle(img, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        }
        return img;
    }

    cv::Mat main_process() {
        /*
	Performs inference using an ONNX model and returns the output image with drawn detections.
	Returns:
		result_image: The output image with drawn detections.
	*/
 
        // Preprocess the image
        cv::Mat blob = this->preprocess();

        // Get input tensor shape info
        size_t tpixels = this->model_input_h * this->model_input_w * 3;
        std::array<int64_t, 4> input_shape_info{1, 3, this->model_input_h, this->model_input_w};

        // Create input tensor
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

        const std::array<const char*, 1> inputNames = {this->input_node_names[0].c_str()};
        const std::array<const char*, 1> outNames = {this->output_node_names[0].c_str()};

        // Perform inference
        std::vector<Ort::Value> ort_outputs;
        try {
            ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());
        } catch (const std::exception& e) {
            std::cerr << "Error during ONNX inference: " << e.what() << std::endl;
            return cv::Mat();
        }

        // Get model output
        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat dout(this->model_output_h, this->model_output_w, CV_32F, (float*)pdata);
        cv::Mat det_output = dout.t();

        // Post-process detections
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; i++) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, 4 + this->labels.size());

            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            if (score > this->confidence_thres) {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);

                cv::Rect box(x, y, width, height);
                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, this->confidence_thres, this->iou_thres, indexes);

        // Draw detections
        this->draw_detections(this->result_image, indexes, boxes, classIds);

        return result_image;
    }
};

int main() {
    // Load the labels and test the class with an image
    std::vector<std::string> labels = {"beet", "corn", "cotton", "pumpkin", "sorghum", "soybean", "spinach", "watermelon", "wheat", "cowpea"};
    std::string img_name = "/home/kingargroo/seed/validate/QYTC20240423666_20240921092551073.jpg";
    std::string onnx_path_name = "/home/kingargroo/seed/ablation1/normal.onnx";

    // Load the image
    cv::Mat input_image = cv::imread(img_name);

    // Instantiate the inference class
    YolovUltralyticsInference inference(labels, onnx_path_name, input_image);

    // Perform inference
    cv::Mat output_image = inference.main_process();

    // Save the output image
    if (!output_image.empty()) {
        cv::imwrite("result.jpg", output_image);
        std::cout << "Inference completed and result saved." << std::endl;
    } else {
        std::cerr << "Inference failed." << std::endl;
    }

    return 0;
}
