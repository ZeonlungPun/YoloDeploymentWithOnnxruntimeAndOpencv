#include <unistd.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>


using namespace cv;
using namespace std;
using namespace Ort;


std::vector<std::string> readLabels(const std::string& labelPath) {
    std::vector<std::string> labels;
    std::ifstream infile(labelPath);
    std::string line;
    while (std::getline(infile, line)) {
        labels.push_back(line);
    }
    infile.close();
    return labels;
}


size_t vectorProduct(const std::vector<int64_t>& vector) {
    if (vector.empty())
        return 0;
    
    size_t product = 1;
    for (const auto& element : vector)
        product *= element;
    
    return product;
}

vector<vector<float>> bbox_cxcywh_to_xyxy(const vector<vector<float>>& boxes)
{
    vector<vector<float>> xyxy_boxes;
    for (const auto& box : boxes)
    {
        float x1 = box[0] - box[2] / 2.0f;
        float y1 = box[1] - box[3] / 2.0f;
        float x2 = box[0] + box[2] / 2.0f;
        float y2 = box[1] + box[3] / 2.0f;
        xyxy_boxes.push_back({ x1, y1, x2, y2 });
    }
    return xyxy_boxes;
}


bool is_normalized(const std::vector<std::vector<float>>& values) {
    for (const auto& row : values) {
        for (const auto& val : row) {
            if (val <= 0 || val >= 1) {
                return false;
            }
        }
    }
    return true;
}

void normalize_scores(std::vector<std::vector<float>>& scores) {
    for (auto& row : scores) {
        for (auto& val : row) {
            val = 1 / (1 + std::exp(-val));
        }
    }
}

vector<vector<int>> generate_class_colors(int num_classes) {
    vector<vector<int>> class_colors(num_classes, vector<int>(3));
    for (int i = 0; i < num_classes; ++i) {
        class_colors[i][0] = rand() % 256;
        class_colors[i][1] = rand() % 256;
        class_colors[i][2] = rand() % 256;
    }
    return class_colors;
}

void draw_boxes_and_save_image(
    const std::vector<int>& labels, 
    const std::vector<float>& scores, 
    const std::vector<std::vector<float>>& boxes, 
    const std::vector<std::string>& CLASS_NAMES,
    cv::Mat& im0
) {
    vector<vector<int>> CLASS_COLORS = generate_class_colors(CLASS_NAMES.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        int label = labels[i];
        float score = scores[i];
        std::ostringstream oss;
        oss << CLASS_NAMES[label] << ": " << std::fixed << std::setprecision(2) << score;
        std::string label_text = oss.str();
        cv::Rect rect((int)boxes[i][0], (int)boxes[i][1], (int)(boxes[i][2] - boxes[i][0]), (int)(boxes[i][3] - boxes[i][1]));
        cv::Scalar color(CLASS_COLORS[label][0], CLASS_COLORS[label][1], CLASS_COLORS[label][2]);
        cv::rectangle(im0, rect, color, 2);
        cv::putText(im0, label_text, cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    
}

cv::Mat MainProcess(size_t deviceId,size_t batchSize,bool useCUDA,cv::Mat imageBGR,std::vector<std::string> labels,std::string mdoelPath,
float confThreshold,std::vector<int> &filtered_labels,std::vector<float>& max_filtered_scores,std::vector<std::vector<float>>& filtered_boxes)
{
    std::string instanceName = "rtdetr-onnxruntime-inference";
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = deviceId;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level [Available levels are as below]
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    // Create session
    Ort::Session ortSession(env, mdoelPath.c_str(), sessionOptions);

    // create allocater ,for allocating memory
    Ort::AllocatorWithDefaultOptions allocator;

    // get input node number
    size_t numInputNodes = ortSession.GetInputCount();
    // get output node number
    size_t numOutputNodes = ortSession.GetOutputCount();

    // get input node and name
    std::vector <std::string> inputNodeNames;
    std::vector <vector <int64_t>> inputNodeDims;
    for (int i = 0; i < numInputNodes; i++) {
        auto inputName = ortSession.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(inputName.get());
        Ort::TypeInfo inputTypeInfo = ortSession.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputDims = inputTensorInfo.GetShape();

        // output value type https://onnxruntime.ai/docs/api/c/group___global.html#gaec63cdda46c29b8183997f38930ce38e
        // return: 1 2nd type(count from zero)， ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT type
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

        if (inputDims.at(0) == -1)
        {
            std::cout << "[Warning] Got dynamic batch size. Setting output batch size to "
                    << batchSize << "." << std::endl;
            inputDims.at(0) = batchSize;
        }

        inputNodeDims.push_back(inputDims);

        std::cout << "[INFO] Input name and shape is: " << inputName.get() << " [";
        for (size_t j = 0; j < inputDims.size(); j++) {
            std::cout << inputDims[j];
            if (j != inputDims.size()-1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }

    // get output node name
    std::vector <std::string> outputNodeNames;
    std::vector <vector <int64_t>> outputNodeDims;
    for (int i = 0; i < numOutputNodes; i++) {
        auto outputName = ortSession.GetOutputNameAllocated(i, allocator);
        outputNodeNames.push_back(outputName.get());
        Ort::TypeInfo outputTypeInfo = ortSession.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();

        if (outputDims.at(0) == -1)
        {
            std::cout << "[Warning] Got dynamic batch size. Setting output batch size to "
                    << batchSize << "." << std::endl;
            outputDims.at(0) = batchSize;
        }

        outputNodeDims.push_back(outputDims);

        std::cout << "[INFO] Output name and shape is: " << outputName.get() << " [";
        for (size_t j = 0; j < outputDims.size(); j++) {
            std::cout << outputDims[j];
            if (j != outputDims.size()-1) {
                std::cout << ",";
            }
        }
        std::cout << ']' << std::endl;
    }
    std::cout << "[INFO] Model was initialized." << std::endl;

    /*       Preprocess     */

    

    // sourec image size
    int64_t imageHeight = imageBGR.rows;
    int64_t imageWidth = imageBGR.cols;
    std::cout << "[INFO] Source image size (h, w) is [" << imageHeight << ", " << imageWidth << "]" << std::endl;

    // model input size
    int64_t inputHeight = inputNodeDims[0].at(2);
    int64_t inputWidth = inputNodeDims[0].at(3);

    // rescale ratio
    float ratioHeight = static_cast<float>(inputHeight) / imageHeight;
    float ratioWidth = static_cast<float>(inputWidth) / imageWidth;

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImageNormRGB, resizedImageNormRGBCHW, preprocessedImage;

    // image resclae
    cv::resize(imageBGR, resizedImageBGR, 
            cv::Size(0, 0), ratioWidth, ratioHeight, cv::INTER_LINEAR);
    std::cout << "[INFO] [Preprocess] Resize" << std::endl;

    // [BGR -> RGB]
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    std::cout << "[INFO] [Preprocess] BGR to RGB" << std::endl;

    // image normalize
    resizedImageRGB.convertTo(resizedImageNormRGB, CV_32FC3, 1.0 / 255);
    std::cout << "[INFO] [Preprocess] Normalization" << std::endl;

    // [HWC -> CHW]
    float *blob = nullptr;
    blob = new float[resizedImageNormRGB.cols * resizedImageNormRGB.rows * resizedImageNormRGB.channels()];
    cv::Size floatImageSize {resizedImageNormRGB.cols, resizedImageNormRGB.rows};
    std::vector<cv::Mat> chw(resizedImageNormRGB.channels());
    for (int i = 0; i < resizedImageNormRGB.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(resizedImageNormRGB, chw);
    std::cout << "[INFO] [Preprocess] HWC to CHW" << std::endl;

    // [CHW -> NCHW]
    std::vector<int64_t> inputTensorShape = {1, 3, inputHeight, inputWidth};

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));
    std::cout << "[INFO] [Preprocess] CHW to NCHW" << std::endl;

    // check input and output nodes
    for (const auto& inputNodeName : inputNodeNames) {
        if (std::string(inputNodeName).empty()) {
            std::cerr << "Empty input node name found." << std::endl;
        }
    }

    // format conversion
    std::vector<const char*> inputNodeNamesCStr;
    for (const auto& inputName : inputNodeNames) {
        inputNodeNamesCStr.push_back(inputName.c_str());
    }
    std::vector<const char*> outputNodeNamesCStr;
    for (const auto& outputName : outputNodeNames) {
        outputNodeNamesCStr.push_back(outputName.c_str());
    }

    /*     Inference       */
    std::vector<Ort::Value> outputTensors = ortSession.Run(
        Ort::RunOptions{nullptr}, 
        inputNodeNamesCStr.data(),
        inputTensors.data(), 
        inputTensors.size(),
        outputNodeNamesCStr.data(),
        1
    );
    std::cout << "[INFO] [Inference] Successfully!" << std::endl;

    /*   Post-Preprocess  */

    // get outputs
    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);
    std::cout << "[INFO] [Postprocess] Get output results" << std::endl;

    // get boxes and scores 
    int num_boxes = outputShape[1];
    int num_classes = labels.size();
    vector<vector<float>> boxes(num_boxes, vector<float>(4));
    vector<vector<float>> scores(num_boxes, vector<float>(num_classes));
    int score_start_index = 4;
    int score_end_index = 4 + num_classes;
    for (int i = 0; i < num_boxes; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            boxes[i][j] = rawOutput[i * score_end_index + j];
        }
        for (int j = score_start_index; j < score_end_index; ++j)
        {
            scores[i][j - score_start_index] = rawOutput[i * score_end_index + j];
        }
    }
    std::cout << "[INFO] [Postprocess] Extract boxes and scores " << std::endl;

    
    vector<vector<float>> xyxy_boxes = bbox_cxcywh_to_xyxy(boxes);

    
    if (!is_normalized(scores)) {
        normalize_scores(scores);
    }

    // get maximun element
    std::vector<float> max_scores;
    for (const auto& score_row : scores) {
        auto max_score = *std::max_element(score_row.begin(), score_row.end());
        max_scores.push_back(max_score);
    }

    // filtering based on scores
    std::vector<bool> mask;
    for (const auto& max_score : max_scores) {
        mask.push_back(max_score > confThreshold);
    }

    // filtering box
    std::vector<std::vector<float>>  filtered_scores;
    for (std::size_t i = 0; i < xyxy_boxes.size(); ++i) {
        if (mask[i]) {
            filtered_boxes.push_back(xyxy_boxes[i]);
            filtered_scores.push_back(scores[i]);
        }
    }

    // get class ID
    for (const auto& score_row : filtered_scores) {
        auto max_score_it = std::max_element(score_row.begin(), score_row.end());
        auto max_score = *max_score_it;
        auto label = std::distance(score_row.begin(), max_score_it);
        filtered_labels.push_back(label);
        max_filtered_scores.push_back(max_score);
    }


    // exctrate  x1, y1, x2, y2
    std::vector<float> \
        x1(filtered_boxes.size()), y1(filtered_boxes.size()), \
        x2(filtered_boxes.size()), y2(filtered_boxes.size());
    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = filtered_boxes[i][0];
        y1[i] = filtered_boxes[i][1];
        x2[i] = filtered_boxes[i][2];
        y2[i] = filtered_boxes[i][3];
    }

    // 对 x1, y1, x2, y2 rescale,adjust,clip
    for (int i = 0; i < filtered_boxes.size(); i++) {
        x1[i] = std::floor(std::min(std::max(1.0f, x1[i] * imageWidth), imageWidth - 1.0f));
        y1[i] = std::floor(std::min(std::max(1.0f, y1[i] * imageHeight), imageHeight - 1.0f));
        x2[i] = std::ceil(std::min(std::max(1.0f, x2[i] * imageWidth), imageWidth - 1.0f));
        y2[i] = std::ceil(std::min(std::max(1.0f, y2[i] * imageHeight), imageHeight - 1.0f));
    }

    // concatnate x1, y1, x2, y2 to boxes
    std::vector<std::vector<float>> new_boxes(filtered_boxes.size(), std::vector<float>(4));
    for (int i = 0; i < filtered_boxes.size(); i++) {
        new_boxes[i][0] = x1[i];
        new_boxes[i][1] = y1[i];
        new_boxes[i][2] = x2[i];
        new_boxes[i][3] = y2[i];
    }
    filtered_boxes = new_boxes;

    return imageBGR;
}

void MainProcessForImage(std::string savePath,size_t deviceId,size_t batchSize,bool useCUDA,std::string imagePath,std::vector<std::string> labels,std::string mdoelPath,
float confThreshold,std::vector<int> &filtered_labels,std::vector<float>& max_filtered_scores,std::vector<std::vector<float>>& filtered_boxes )
{
    // read image
    cv::Mat imageBGR = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
    imageBGR=MainProcess(deviceId, batchSize,useCUDA,imageBGR, labels,mdoelPath,confThreshold, filtered_labels,max_filtered_scores,filtered_boxes);

    

    // draw the results and save
    draw_boxes_and_save_image(
        filtered_labels,
        max_filtered_scores,
        filtered_boxes,
        labels,
        imageBGR
    );
    cv::imwrite(savePath, imageBGR);
    std::cout << "[INFO] [Postprocess] Done! " << std::endl;
}
void MainProcessForVideo(std::string savePath,size_t deviceId,size_t batchSize,bool useCUDA,std::string VideoPath,std::vector<std::string> labels,std::string mdoelPath,
float confThreshold,std::vector<int> &filtered_labels,std::vector<float>& max_filtered_scores,std::vector<std::vector<float>>& filtered_boxes)
{
	
    
    VideoCapture cap(VideoPath);
	//VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "load error" << endl;
	}
	Mat frame;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		int64 start = cv::getTickCount();
		frame =MainProcess(deviceId, batchSize,useCUDA,frame, labels,mdoelPath,confThreshold, filtered_labels,max_filtered_scores,filtered_boxes);
        // draw the results and save
        draw_boxes_and_save_image(filtered_labels, max_filtered_scores, filtered_boxes,labels,frame);

		// FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("rtdert",frame);
		cv::waitKey(10);

	}


}

int main(int argc, char* argv[])
{

    // check work path
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("[INFO] Current working directory is: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Note: There is not a C++ API that returns ORT version. 
    // Only C, so you shold include <onnxruntime_c_api.h>
    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    
    bool useCUDA = false;
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";

    if (argc == 1) {
        useCUDA = false;
    }
    else if ( (argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0) ) {
        useCUDA = true;
    }
    else if ( (argc == 2) && (strcmp(argv[1], useCPUFlag) == 0) ) {
        useCUDA = false;
    }
    else {
        throw std::runtime_error("Invalid #Param, please check double again!");
    }

    if (useCUDA) {
        std::cout << "[INFO] Inference execution provider: CUDA" << std::endl;
    }
    else {
        std::cout << "[INFO] Inference execution provider: CPU" << std::endl;
    }

    std::string imagePath = "/home/kingargroo/YOLOVISION/beaty.jpg";
    std::string savePath = "/home/kingargroo/YOLOVISION/beaty1.jpg";
    std::string labelPath = "/home/kingargroo/cpp/yolov8onnx/classes.txt";
    std::string mdoelPath = "/home/kingargroo/YOLOVISION/rtdetr-l.onnx";
    
    size_t deviceId = 0;
    size_t batchSize = 1;
    float confThreshold = 0.45;
    
    std::vector<std::string> labels = readLabels(labelPath);
    
    if (labels.empty()) {
        throw std::runtime_error("No labels found!");
    }
    std::vector<int>filtered_labels;
    std::vector<float>max_filtered_scores;
    std::vector<std::vector<float>> filtered_boxes;
    std::string videoPath="/home/kingargroo/YOLOVISION/people.mp4";
    //MainProcessForImage(savePath, deviceId,batchSize,useCUDA, imagePath,labels, mdoelPath,confThreshold,filtered_labels, max_filtered_scores, filtered_boxes);
    MainProcessForVideo(savePath, deviceId,batchSize,useCUDA, videoPath,labels, mdoelPath,confThreshold,filtered_labels, max_filtered_scores, filtered_boxes);

}