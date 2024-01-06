#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include <opencv2/opencv.hpp>
#include <fstream>

std::string labels_txt_file = "E:\\opencv\\YOLO\\coco.names";
std::vector<std::string> readClassNames();
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

int main(int argc, char** argv) {
	float x_factor = 0.0;
	float y_factor = 0.0;

	cv::RNG rng;
	std::vector<cv::Scalar> color_tables;
	for (int i = 0; i < 17; i++) {
		int a = rng.uniform(0, 255);
		int b = rng.uniform(0, 255);
		int c = rng.uniform(0, 255);
		color_tables.push_back(cv::Scalar(a, b, c));
	}

	std::vector<std::string> labels = readClassNames();
	cv::Mat frame = cv::imread("E:\\opencv\\tracy35.jpg");
	//     InferSession,
	// GPU Mode, 0 - gpu device id
	std::string onnxpath = "D:\\deploy\\models\\yolov8n-pose.onnx";
	std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8n-pose-onnx");

	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	//std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
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

	// query output data format
	int out_h = 0; // 56= box 4+conf1 + 17x3 17個關鍵點 x,y,z(表示是否可見：1表示可見，0表示不可見)
	int out_w = 0; // 8400
	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		out_h = outShapeInfo[1];
		out_w = outShapeInfo[2];
		std::cout << "output format: " << out_h << "x" << out_w << std::endl;
	}

	
	int64 start = cv::getTickCount();
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	//模型是將圖片resize到（640,640）進行推斷的，故模型結果要與一個系數作乘法以適應圖片真正大小
	x_factor = image.cols / static_cast<float>(640);
	y_factor = image.rows / static_cast<float>(640);

	//用於存放關鍵點結果的matrix,初始化時保持縮放圖像縮放因子
	cv::Mat keypoints_mat = cv::Mat::zeros(cv::Size(3, 17), CV_32FC1);
	for (int i = 0; i < 17; i++) {
		keypoints_mat.at<float>(i, 0) = x_factor;
		keypoints_mat.at<float>(i, 1) = y_factor;
		keypoints_mat.at<float>(i, 2) = 1.0f;
	}

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
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
	// 56x84
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> multiple_kypts;
	std::vector<float> confidences;
	cv::Mat dout(out_h, out_w, CV_32F, (float*)pdata);

	cv::Mat det_output = dout.t(); // 56x8400 => 8400x56
	for (int i = 0; i < det_output.rows; i++) {
		double score = det_output.at<float>(i, 4);
		// 確認檢測到人
		if (score > 0.85)
		{
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
			confidences.push_back(score);
			cv::Mat pts = det_output.row(i).colRange(5, out_h);
			multiple_kypts.push_back(pts);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

	// show boxes with objects

	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2, 8, 0);
		putText(frame, "person", boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
		cv::Mat one_kypts = multiple_kypts[idx];
		std::cout << one_kypts << std::endl;
		cv::Mat  one_kypt = one_kypts.reshape(0, 17);
		cv::Mat kpts; // 17x3
		cv::multiply(one_kypt, keypoints_mat, kpts);

		//把所有的點連接起來（畫連接線）
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), color_tables[0], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(1, 0), kpts.at<float>(1, 1)), cv::Point(kpts.at<float>(3, 0), kpts.at<float>(3, 1)), color_tables[1], 2, 8, 0);

		// nose->right_eye->right_ear.(0, 2), (2, 4)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), color_tables[2], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(2, 0), kpts.at<float>(2, 1)), cv::Point(kpts.at<float>(4, 0), kpts.at<float>(4, 1)), color_tables[3], 2, 8, 0);

		// nose->left_shoulder->left_elbow->left_wrist.(0, 5), (5, 7), (7, 9)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), color_tables[4], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), color_tables[5], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(7, 0), kpts.at<float>(7, 1)), cv::Point(kpts.at<float>(9, 0), kpts.at<float>(9, 1)), color_tables[6], 2, 8, 0);

		// nose->right_shoulder->right_elbow->right_wrist.(0, 6), (6, 8), (8, 10)
		cv::line(frame, cv::Point(kpts.at<float>(0, 0), kpts.at<float>(0, 1)), cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), color_tables[7], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), color_tables[8], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(8, 0), kpts.at<float>(8, 1)), cv::Point(kpts.at<float>(10, 0), kpts.at<float>(10, 1)), color_tables[9], 2, 8, 0);

		// left_shoulder->left_hip->left_knee->left_ankle.(5, 11), (11, 13), (13, 15)
		cv::line(frame, cv::Point(kpts.at<float>(5, 0), kpts.at<float>(5, 1)), cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), color_tables[10], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(11, 0), kpts.at<float>(11, 1)), cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), color_tables[11], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(13, 0), kpts.at<float>(13, 1)), cv::Point(kpts.at<float>(15, 0), kpts.at<float>(15, 1)), color_tables[12], 2, 8, 0);

		// right_shoulder->right_hip->right_knee->right_ankle.(6, 12), (12, 14), (14, 16)
		cv::line(frame, cv::Point(kpts.at<float>(6, 0), kpts.at<float>(6, 1)), cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), color_tables[13], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(12, 0), kpts.at<float>(12, 1)), cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), color_tables[14], 2, 8, 0);
		cv::line(frame, cv::Point(kpts.at<float>(14, 0), kpts.at<float>(14, 1)), cv::Point(kpts.at<float>(16, 0), kpts.at<float>(16, 1)), color_tables[16], 2, 8, 0);

		//畫檢測到的關鍵點
		for (int row = 0; row < kpts.rows; row++) {
			int x = static_cast<int>(kpts.at<float>(row, 0));
			int y = static_cast<int>(kpts.at<float>(row, 1));
			cv::circle(frame, cv::Size(x, y), 3, cv::Scalar(255, 0, 255), 4, 8, 0);
		}
	}
	//     FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

	cv::imshow("ONNXRUNTIME1.13 + YOLOv8 Pose       ʾ", frame);
	cv::waitKey(0);

	// relase resource
	session_options.release();
	session_.release();
	return 0;
}