#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;
using namespace dnn;

std::vector<std::string> class_names{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


void porcess_with_one_image(std::string& onnxpath,cv::Mat& frame,float modelinput_height, float modelinput_width)
{
    float modelScoreThreshold=0.2;
    float modelNMSThreshold=0.3;
    // load the onnx model 
    auto net = cv::dnn::readNetFromONNX(onnxpath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    //image preprocess
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    float x_factor = image.cols / modelinput_width;
    float y_factor = image.rows / modelinput_height;

    // inference using model
    cv::Mat blob;
    cv::dnn::blobFromImage(image,blob,1 / 255.0, cv::Size(modelinput_height, modelinput_width), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // postprocess the image 
    // yolov8 has an output of shape (batchSize, 84,  8400) ( box[x,y,w,h]+Num classes )
    int rows = outputs[0].size[2]; //8400
    int dimensions =outputs[0].size[1]; //84
    std::cout<<"rows is:"<<rows<<std::endl;
    std::cout<<"dim is:"<<dimensions<<std::endl;

    //(1,84,8400)-->(84,8400)
    outputs[0] = outputs[0].reshape(1, dimensions);
    //(84,8400) --> (8400,84)
    cv::transpose(outputs[0], outputs[0]);

    //get the first data pointer;finaaly will get all 8400 bounding box
    float *data = (float *)outputs[0].data;

    // the storage to save the results
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    //rows stand for the total number of bounding box
    for (int i = 0; i < rows; ++i)
    {
        
        //skip the x,y,w,h 
        float *classes_scores = data+4;

        cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);
            

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        
        
        //pointer to next 84  group
        data += dimensions;
        
    }


    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = class_ids[index];
        cv::rectangle(frame, boxes[index], (0,0,255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
            cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(255, 255, 255), -1);
        cv::putText(frame, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
    }


}

void detect_image(std::string& onnxpath,cv::Mat& frame,float modelinput_height, float modelinput_width)
{
    int64 start = cv::getTickCount();
    porcess_with_one_image(onnxpath,frame,modelinput_height, modelinput_width);
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);



}

void detect_video(std::string& onnxpath,float modelinput_height, float modelinput_width,std::string video_name)
{
    
    cv::VideoCapture capture(video_name);
    cv::Mat frame;
    while (true) 
    {
        bool ret = capture.read(frame);
        if (frame.empty()) {
            break;
        }
        int64 start = cv::getTickCount();
        porcess_with_one_image(onnxpath,frame,modelinput_height, modelinput_width);
        float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("OpenCV4.8 inference FOR YOLOV8", frame);

        char c = cv::waitKey(1);
        if (c == 27) 
        {
            break;
        }

    }

}



int main()
{
    std::string onnxpath = "/home/punzeonlung/CPP/yolov8/yolov8m.onnx";
    cv::Mat frame=cv::imread("/home/punzeonlung/CPP/yolov8cpp/test5.png");
    // detect_image(onnxpath,frame,1280.0, 1280.0);
    // cv::imshow("OpenCV4.8 inference FOR YOLOV8", frame);
    // char c = cv::waitKey(0);
    detect_video(onnxpath,1280,1280,"/home/punzeonlung/CPP/ByteTrack/people.mp4");
    

    return 0;
}





 
   

    
   

    