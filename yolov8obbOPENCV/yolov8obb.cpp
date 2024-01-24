#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

// defind some constant variables
#define pi acos(-1)
float modelScoreThreshold=0.3;
float modelNMSThreshold=0.35;
std::vector<std::string> classes{"plane","ship","storage tank","baseball diamond","tennis court" ,"basketball court","ground track field","harbor","bridge","large vehicle","small vehicle","helicopter","roundabout","soccer ball field","swimming pool"};


// define a struct to save some information
typedef struct {
	cv::RotatedRect box;
	float score;
	int Classindex;
}RotatedBOX;


cv::Mat formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<RotatedBOX> detect_one_image(const cv::Mat &input_image,std::vector<std::string>& classes,cv::dnn::Net& net)
{
    cv::Mat modelInput = input_image;
    modelInput = formatToSquare(modelInput);
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, cv::Size {640, 640}, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

   

    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];
    cout<<"r:"<<rows<<endl;
    cout<<"d:"<<dimensions<<endl;
    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);
    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / 640.0;
    float y_factor = modelInput.rows / 640.0;

   
    std::vector<cv::RotatedRect> boxes;
    std::vector<RotatedBOX>BOXES;
    std::vector<float> confidences;

    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data+4;
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        RotatedBOX BOX;
        
        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);  
            float x = data[0]* x_factor;
            float y = data[1]* y_factor;
            float w = data[2]* x_factor;
            float h = data[3]* y_factor;

            float angle=data[dimensions-1];
            //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
            if (angle>=pi && angle <= 0.75*pi)
            {
                angle=angle-pi;
            } 

            BOX.Classindex=class_id.x;
            BOX.score=maxClassScore; 
            cv::RotatedRect box=cv::RotatedRect(cv::Point2f(x,y),cv::Size2f(w,h),angle*180/pi);
            BOX.box=box;
            boxes.push_back(box);
            BOXES.push_back(BOX);
                
        }
        data += dimensions;
    }
    cout<<boxes.size()<<endl;
    //postprocess
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    cout<<nms_result.size()<<endl;
    std::vector<RotatedBOX> Remain_boxes;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        RotatedBOX Box_=BOXES[idx];
        Remain_boxes.push_back(Box_);
    }

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

        // 將旋轉矩形畫到圖像上
        for (int i = 0; i < 4; ++i) 
        {
            cv::line(image, points[i], points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);  
        }
        cv::putText(image, classes[class_id], points[0], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 2);


    }

     // 顯示圖像
    cv::imshow("RotatedRect", image);
    cv::waitKey(0);


}


int main()
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/punzeonlung/CPP/yolov8-obb/yolov8n-obb.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::Mat input_image=cv::imread("/home/punzeonlung/CPP/yolov8-obb/test1.jpg");
    int64 start = cv::getTickCount();
    std::vector<RotatedBOX>detect_boxes = detect_one_image(input_image, classes, net);
    Draw( input_image, detect_boxes);
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(input_image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);


}