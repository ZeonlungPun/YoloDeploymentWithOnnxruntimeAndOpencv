#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>

#define pi acos(-1)
using namespace std;
using namespace cv;

float modelScoreThreshold=0.3;
float modelNMSThreshold=0.35;
std::vector<std::string> classes={"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench","bird"};



cv::Mat formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<cv::RotatedRect> detect_one_image(const cv::Mat &input_image,std::vector<std::string>& classes,cv::dnn::Net& net)
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

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::RotatedRect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data+4;
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        
        if (maxClassScore > modelScoreThreshold)
        {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
              
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
                
             

                cv::RotatedRect box=cv::RotatedRect(cv::Point2f(x,y),cv::Size2f(w,h),angle*180/pi);
                boxes.push_back(box);
                
        }
        data += dimensions;
    }
    cout<<boxes.size()<<endl;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    cout<<nms_result.size()<<endl;
    std::vector<cv::RotatedRect> Remain_boxes;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        cv::RotatedRect box_=boxes[idx];
        Remain_boxes.push_back(box_);
    }

    return Remain_boxes;
    
}

void Draw(cv::Mat& image,std::vector<cv::RotatedRect>& detect_boxes)
{
    for (unsigned long i = 0; i < detect_boxes.size(); ++i)
    {
       
        cv::RotatedRect box_=detect_boxes[i];
        cv::Point2f points[4];
        box_.points(points);
        // 將旋轉矩形畫到圖像上
        for (int i = 0; i < 4; ++i) 
        {
            cv::line(image, points[i], points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);  // 顏色為紅色，線寬為2
        }

    }
     // 顯示圖像
    cv::imshow("RotatedRect", image);
    cv::waitKey(0);


}


int main()
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/kingargroo/cpp/yolov8/yolov8n-obb.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::Mat input_image=cv::imread("/home/kingargroo/cpp/yolov8/test1.jpeg");
    std::vector<cv::RotatedRect>detect_boxes = detect_one_image(input_image, classes, net);
    Draw( input_image, detect_boxes);

}