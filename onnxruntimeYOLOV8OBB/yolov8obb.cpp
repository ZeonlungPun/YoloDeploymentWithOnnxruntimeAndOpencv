#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
#define pi acos(-1)
float modelScoreThreshold=0.3;
float modelNMSThreshold=0.35;

std::vector<std::string> labels = {"plane","ship","storage tank","baseball diamond","tennis court" ,"basketball court","ground track field","harbor","bridge","large vehicle","small vehicle","helicopter","roundabout","soccer ball field","swimming pool"};


cv::Mat formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

// define a struct to save some information
typedef struct {
	cv::RotatedRect box;
	float score;
	int Classindex;
}RotatedBOX;

std::vector<RotatedBOX> main_detectprocess_with_yolov8(std::string& onnx_path_name, std::vector<std::string>& labels,Mat &frame)
{
	
	// format frame

    cv::Mat image=formatToSquare(frame);

	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	//std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	Ort::Session session_(env, onnx_path_name.c_str(), session_options);

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// get the input information
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

	// get the output information
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 84
	output_w = output_dims[2]; // 8400

	std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

	
	float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);

	cv::Mat blob;
    cv::dnn::blobFromImage(image,blob, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	// output data
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// post-process
	std::vector<cv::RotatedRect> boxes;
	std::vector<float> confidences;
    std::vector<RotatedBOX>BOXES;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        RotatedBOX BOX;

		// confidence between 0 and 1
		if (score > modelScoreThreshold)
		{
			float cx = det_output.at<float>(i, 0)*x_factor;
			float cy = det_output.at<float>(i, 1)* y_factor;
			float ow = det_output.at<float>(i, 2)*x_factor;
			float oh = det_output.at<float>(i, 3)* y_factor;
            float angle=det_output.at<float>(i,4+labels.size());
            //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
            if (angle>=pi && angle <= 0.75*pi)
            {
                angle=angle-pi;
            }
           
            BOX.Classindex=classIdPoint.x;
            BOX.score=score; 
            cv::RotatedRect box=cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
            BOX.box=box;
            boxes.push_back(box);
            BOXES.push_back(BOX); 
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences,modelScoreThreshold,modelNMSThreshold, nms_result);
	std::vector<RotatedBOX> Remain_boxes;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        RotatedBOX Box_=BOXES[idx];
        Remain_boxes.push_back(Box_);
    }

    
	session_options.release();
	session_.release();
    return Remain_boxes;

}

void Draw(cv::Mat& image,std::vector<RotatedBOX>& detect_boxes)
{
    for (unsigned long i = 0; i < detect_boxes.size(); ++i)
    {
       
        RotatedBOX Box_=detect_boxes[i];
        cv::RotatedRect box_=Box_.box;
        cv::Point2f points[4];
        box_.points(points);
        int class_id=Box_.Classindex;

        for (int i = 0; i < 4; ++i) 
        {
            cv::line(image, points[i], points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);  
        }
        cv::putText(image, labels[class_id], points[0], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 2);


    }

    cv::imshow("RotatedRect", image);
    cv::waitKey(0);


}


int main()
{
    std::string model_path="/home/kingargroo/cpp/yolov8obbOPENCV/yolov8n-obb.onnx";
    cv::Mat input_image=cv::imread("/home/kingargroo/cpp/yolov8obbOPENCV/test4.jpg");
    std::vector<RotatedBOX> detect_boxes=main_detectprocess_with_yolov8(model_path,  labels,input_image);
    Draw( input_image, detect_boxes);
    


}