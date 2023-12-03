#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include <opencv2/opencv.hpp>
#include <fstream>

std::string labels_txt_file = "E:\\opencv\\YOLO\\coco.names";
std::vector<std::string> readClassNames();
float sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

cv::Mat main_detectprocess_with_yolov8(std::string& onnxpath, std::vector<std::string>& labels, cv::Mat& frame,float& imgsize)
{
	cv::RNG rng;
	float x_factor = 0.0;
	float y_factor = 0.0;
	//最終輸出的mask與標準輸入尺寸的比值
	float sx = 160.0f / imgsize;
	float sy = 160.0f / imgsize;
	// InferSession
	std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8seg-onnx");

	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
	OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
	Ort::Session session_(env, modelPath.c_str(), session_options);

	// get input and output info
	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();
	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;
	Ort::AllocatorWithDefaultOptions allocator;
	int input_h = 0;
	int input_w = 0;

	// query input data format
	for (int i = 0; i < input_nodes_num; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int ch = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
	}

	// query output data format   25200x85
	int output_h1 = 0;
	int output_w1 = 0;
	int output_h2 = 0;
	int output_w2 = 0;
	int output_c = 0;
	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		if (i == 0)
		{
			output_h1 = outShapeInfo[1]; // 116
			output_w1 = outShapeInfo[2]; // 8400
			std::cout << "output format" << i << ":" << output_h1 << "x" << output_w1 << std::endl;
		}
		else
		{
			output_c = outShapeInfo[1]; // k=32
			output_h2 = outShapeInfo[2]; // 160 prototype h
			output_w2 = outShapeInfo[3]; // 160  prototype w
			std::cout << "output format" << i << ":" << output_c << "x" << output_h2 << "x" << output_w2 << std::endl;
		}

	}

	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	x_factor = image.cols / static_cast<float>(imgsize);
	y_factor = image.rows / static_cast<float>(imgsize);

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	//parallel output
	const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };

	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
	// 116x8400, 1x32x160x160
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	const float* mdata = ort_outputs[1].GetTensorMutableData<float>();

	//out1: 1x116x8400             out2:1x32x160x160
	//      1xoutput_h1xoutput_w1       1xoutput_cxoutput_h2xoutput_w2
	// 8400=80x80+40x40+20x20  116=box4+num_classes80+32
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks_coefficients; // n (instance number) x k=32
	cv::Mat dout(output_h1, output_w1, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 116x8400 => 8400x116
	cv::Mat prototypes(output_c, output_h2 * output_w2, CV_32F, (float*)mdata); //32x25600  kxhxw

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, output_h1 - 32);
		cv::Point classId;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classId);


		if (score > 0.25)
		{
			//Mask Coefficients 
			cv::Mat masks_coefficient = det_output.row(i).colRange(output_h1 - 32, output_h1);
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			//將圖片由標準輸入尺寸（如640x640）轉化到真實尺寸
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
			classIds.push_back(classId.x);
			confidences.push_back(score);
			masks_coefficients.push_back(masks_coefficient);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());

	//遍歷每個box，獲取box內的instance segement
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		int cid = classIds[idx];
		cv::Rect box = boxes[idx];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::max(0, box.br().x);
		int y2 = std::max(0, box.br().y);
		cv::Mat coef = masks_coefficients[idx];
		//  1 x k=32  32x25600 --> 1x25600
		//a single matrix multiplication and sigmoid
		cv::Mat m = coef * prototypes;
		for (int col = 0; col < m.cols; col++) {
			m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
		}
		// 1x25600 -->  1x160x160
		cv::Mat m1 = m.reshape(1, 160);

		// x1/x_factor : 將圖片由真實尺寸轉化到標準輸入尺寸（如640x640）
		// （x1/x_factor）*sx：將圖片由標準輸入尺寸（如640x640）轉化到 特徵圖（160x160）對應的尺寸
		int mx1 = std::max(0, int((x1 * sx) / x_factor));
		int mx2 = std::max(0, int((x2 * sx) / x_factor));
		int my1 = std::max(0, int((y1 * sy) / y_factor));
		int my2 = std::max(0, int((y2 * sy) / y_factor));

		// fix out of range box boundary 
		if (mx2 >= m1.cols) {
			mx2 = m1.cols - 1;
		}
		if (my2 >= m1.rows) {
			my2 = m1.rows - 1;
		}
		// end fix it!!

		//得到box所在的子圖
		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		//得到像素的分類結果
		for (int r = 0; r < rm.rows; r++) {
			for (int c = 0; c < rm.cols; c++) {
				float pv = rm.at<float>(r, c);
				if (pv > 0.5) {
					rm.at<float>(r, c) = 1.0;
				}
				else {
					rm.at<float>(r, c) = 0.0;
				}
			}
		}
		//任意分配個顏色
		rm = rm * rng.uniform(0, 255);
		rm.convertTo(det_mask, CV_8UC1);
		if ((y1 + det_mask.rows) >= frame.rows) {
			y2 = frame.rows - 1;
		}
		if ((x1 + det_mask.cols) >= frame.cols) {
			x2 = frame.cols - 1;
		}
		//創建一張空白mask
		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		//將目標mask（與box一樣大）複製到空白mask（與原始圖片所在位置）box所在位置
		det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		//將mask與rgbmask相加（融合）
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
		cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2, 8, 0);
		putText(frame, labels[cid].c_str(), boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 4, 8);
	}
	// relase resource
	session_options.release();
	session_.release();
	return rgb_mask;
}

void SegementImage(std::string& onnxpath, std::vector<std::string>& labels,float& imgsize, std::string& img_path)
{
	
	cv::Mat frame = cv::imread(img_path);
	int64 start = cv::getTickCount();
	cv::Mat rgb_mask=main_detectprocess_with_yolov8(onnxpath, labels,  frame,  imgsize);
	//FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

	cv::Mat result;
	//原圖與mask相加
	cv::addWeighted(frame, 0.5, rgb_mask, 0.5, 0, result);
	result.copyTo(frame);

	cv::imshow("ONNXRUNTIME1.13 + YOLOv8-seg", frame);
	cv::waitKey(0);
}

void SegementVideo(std::string& onnxpath, std::vector<std::string>& labels, float& imgsize, std::string& video_path){

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened())
	{
		std::cout << "load error" << std::endl;
	}
	cv::Mat frame;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		int64 start = cv::getTickCount();
		cv::Mat rgb_mask = main_detectprocess_with_yolov8(onnxpath, labels, frame, imgsize);
		// FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		
		cv::Mat result;
		cv::addWeighted(frame, 0.5, rgb_mask, 0.5, 0, result);
		result.copyTo(frame);
		
		cv::imshow("YOLOv8+ONNXRUNTIME ", frame);
		cv::waitKey(10);

	}
}

int main(int argc, char** argv) {
	
	std::vector<std::string> labels = readClassNames();
	//std::string img_path ="E:\\opencv\\tracy35.jpg";
	std::string video_path = "E:\\opencv\\yolo8Sort\\Videos\\motorbikes.mp4";
	std::string onnxpath = "E:\\opencv\\YOLO\\yolov8s-seg.onnx";
	float imgsize = 640;
	//SegementImage( onnxpath, labels,  imgsize,  img_path);
	SegementVideo(onnxpath, labels, imgsize, video_path);

	return 0;
	
	

	
	

}