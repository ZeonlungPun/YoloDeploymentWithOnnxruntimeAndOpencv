#include <opencv2/opencv.hpp>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


class Yolov8InferenceLibtorch
{
public:
    int model_input_height;
    int model_input_wdith;
    std::vector<std::string>labels;
    
   
    float conf_thres;
    float iou_thres;
    int top,left;
    float ratio;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::string save_path;
    std::string model_path;
    std::string labelPath_str;
    torch::Device device;
    torch::jit::script::Module yolo_model;

    Yolov8InferenceLibtorch(
    int model_input_height,
    int model_input_wdith,
    std::string save_path,
    std::string model_path,
    std::string labelPath_str,
    float conf_thres=0.25,
    float iou_thres=0.5):model_input_height(model_input_height),
    model_input_wdith(model_input_wdith),save_path(save_path),
    model_path(model_path),labelPath_str(labelPath_str),conf_thres(conf_thres),
    iou_thres(iou_thres),
    device(torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU))
    {   
        
        this->ReadLabelsfromTxt();
        // Load the model (e.g. yolov8s.torchscript)
        this->yolo_model = torch::jit::load(this->model_path);

    }

    
    void ReadLabelsfromTxt()
    {
        
        // 打開文件
        std::ifstream labelFile(this->labelPath_str);
        std::string line;
        //按行讀取類別
        while (std::getline(labelFile, line))
        {
            if (line.length())
            {
                this->labels.push_back(line);
            }
        }
    }

    cv::Mat preprocess(cv::Mat input_image)
    {
        //lettter_box method
        cv::Mat frame;
        cv::cvtColor(input_image, frame, cv::COLOR_BGR2RGB);
        float ratio = std::min(static_cast<float>(this->model_input_height) / frame.rows,
                            static_cast<float>(this->model_input_wdith) / frame.cols);
        int newh=(int) std::round(frame.rows*ratio);
        int neww=(int) std::round(frame.cols*ratio);
        cv::Size new_unpad(neww,newh);
        //get the padding length in each size
        float dw=(this->model_input_wdith-neww)/2;
        float dh=(this->model_input_height-newh)/2;

        if (neww !=this->model_input_wdith || newh !=this->model_input_height)
        {  //resize the image with same ratio for wdith and height
            cv::resize(frame,frame,new_unpad,cv::INTER_LINEAR);
        }
        // calculate the padding pixel around
        int top,bottom,left,right;
        top =(int) std::round(dh-0.1);
        bottom= (int) std::round(dh+0.1);
        left = (int) std::round(dw-0.1);
        right= (int) std::round(dw+0.1);
        this->top=top;
        this->left=left;
        this->ratio=ratio;


        cv::copyMakeBorder(frame, frame, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return frame;
    }

    std::vector<int> DeconstructPredictionResults(torch::Tensor& predictions) {
        //predictions: (bs=1,output_dim=4+number_classes,num_anchors=8400)
        auto bs = predictions.size(0);
        //number of classes
        auto nc = predictions.size(1) - 4;
        //number of anchors
        auto na = predictions.size(2);
        // (bs,output_dim,num_anchors)-->(bs,num_anchors,output_dim)
        predictions = predictions.transpose(-1, -2);
        predictions = predictions.squeeze(0);
        
        
    
        // Iterate through each prediction
        for (int xi = 0; xi < na; xi++) {

            auto prediction = predictions[xi];
        
            // Split box coordinates and class scores
            auto prediction_split = prediction.split({4, nc}, 0);
            auto box = prediction_split[0];
            auto cls = prediction_split[1];
            

            // Find the max probabilities and corresponding class IDs
            auto [conf, j] = cls.max(0, true);

            // Extract box coordinates
            int cx = static_cast<int>(box[0].item<float>());
            int cy = static_cast<int>(box[1].item<float>());
            int ow = static_cast<int>(box[2].item<float>());
            int oh = static_cast<int>(box[3].item<float>());

            cx = (cx - this->left) / this->ratio;
            cy = (cy - this->top) / this->ratio;
            ow = ow / this->ratio;
            oh = oh / this->ratio;

            // Extract confidence score and class ID
            float score = conf.item<float>();
            int classId = j.item<int>();

            // Add to confidences and classIds
            this->confidences.push_back(score);
            this->classIds.push_back(classId);

            // Calculate bounding box coordinates
            int x = static_cast<int>(cx - 0.5 * ow);
            int y = static_cast<int>(cy - 0.5 * oh);
            int width = static_cast<int>(ow );
            int height = static_cast<int>(oh );

            // Create and store the bounding box
            cv::Rect rect;
            rect.x = x;
            rect.y = y;
            rect.width = width;
            rect.height = height;
            this->boxes.push_back(rect);
        }

        // Perform Non-Maximum Suppression (NMS)
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, this->conf_thres, this->iou_thres, indexes);
        return indexes;

    }

    cv::Mat VisualizeResults(std::vector<int> indexes,cv::Mat frame)
    {
        // Show the results
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int idx = this->classIds[index];
            cv::rectangle(frame, this->boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		    putText(frame, this->labels[idx], cv::Point(this->boxes[index].tl().x, this->boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
            
	    }
        return frame;

    }

    void MainProcessForSingleImage(torch::jit::script::Module yolo_model,torch::Device device,cv::Mat image,int model_input_height,int model_input_wdith)
    {
        cv::Mat model_input_image=preprocess(image);
        yolo_model.eval();
        yolo_model.to(device, torch::kFloat32);
        torch::Tensor image_tensor = torch::from_blob(model_input_image.data, {model_input_image.rows, model_input_image.cols, 3}, torch::kByte).to(device);
        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        // channel first for torch package
        image_tensor = image_tensor.permute({2, 0, 1});
        // (1,C,H,W)
        image_tensor = image_tensor.unsqueeze(0);
        std::vector<torch::jit::IValue> inputs {image_tensor};
        
        // Inference：(bs=1,output_dim=4+number_classes,num_anchors)
        torch::Tensor output = yolo_model.forward(inputs).toTensor().cpu();
        
        std::vector<int> indexes=DeconstructPredictionResults(output);
        cv::Mat output_image=VisualizeResults(indexes,image);

    
    }
    void RunForImage(std::string image_path)
    {
        // Load image and preprocess
        cv::Mat image = cv::imread(image_path);
        MainProcessForSingleImage(yolo_model,this->device, image, this->model_input_height,this->model_input_wdith);
        cv::imwrite(this->save_path,image);
    
    }

    void RunForVideo(std::string video_path)
    {
        cv::VideoCapture cap(video_path);
            
        if (!cap.isOpened())
        {
            std::cout << "load error" <<std::endl;
        }
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int frame_fps = cap.get(cv::CAP_PROP_FPS);

        cv::VideoWriter video(save_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), frame_fps, cv::Size(frame_width, frame_height));

        cv::Mat frame;
        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                break;
            }
            int64 start = cv::getTickCount();
            MainProcessForSingleImage(yolo_model,this->device, frame, this->model_input_height,this->model_input_wdith);
            // FPS render it
            float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
            putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
            video.write(frame);
            

        }
    }

};











int main() {
    
    try {
        
        std::string model_path = "/home/kingargroo/YOLOVISION/yolov8n.torchscript";
        std::string image_path="/home/kingargroo/YOLOVISION/beauty.jpg";
        
        int model_input_height=640;
        int model_input_wdith=640;
        std::string save_path="result.png";
        std::string labelPath_str="/home/kingargroo/cpp/torch1/cocolabels.txt";
        float conf_thres=0.25;
        float iou_thres=0.5;
        Yolov8InferenceLibtorch Inference_class( model_input_height,model_input_wdith,save_path,model_path,labelPath_str,conf_thres,iou_thres);
        Inference_class.RunForImage(image_path);
        
       
    } catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }

    return 0;
}