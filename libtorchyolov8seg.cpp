/*
 * Yolov8segInference: A YOLOV8-seg inference class for instant segmentation
 * 
 * Author: Zeonlung Pun
 * Date: Feb 10, 2025
 * 
 * Description:
 * This class provides an implementation of YOLOV8-seg algorithms inference 
 * using libtorch and OpenCV.
 * It includes methods for reading the model, preprocessing input images, running inference,
 * and drawing  results on the input image.
 * 
 * Usage:
 * The  class is instantiated with the .torchsript model path,
 * input image, and  preprocessing and postprocessing.
 * The `main_process` function performs the complete inference and returns the processed image.
 *
 * Note:
 * This code assumes the libtorch and OpenCV libraries are properly installed and configured.
 * 
 * Contact: zeonlungpun@gmail.com
 * 
 */

#include<stdio.h>
#include<string>
#include<iostream>
#include<vector>
#include<tuple>
#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>
#include<cassert>

float sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}
class Yolov8segInference
{
public:

    std::string model_path;
	torch::jit::script::Module module;
	cv::Mat raw_img;
	int model_input_h;
	int model_input_w;
	int top,left;
    float ratio;
	int ori_w; 
	int ori_h;
	int num_classes;
    float conf_thres;
    float iou_thres;
    // the ratio of mask and standard model input size
    float sx,sy;
	std::string result_save_path;
    // to save the box and mask result
    std::vector<cv::Mat>mask_result_list;
    std::vector<cv::Rect>box_result_list;

    Yolov8segInference
    (std::string model_path,cv::Mat raw_img,
	int model_input_h,int model_input_w,std::string result_save_path,int num_classes,float conf_thres,float iou_thres):
	model_path(model_path),raw_img(raw_img),model_input_h(model_input_h),
	model_input_w(model_input_w),result_save_path(result_save_path),num_classes(num_classes),conf_thres(conf_thres),iou_thres(iou_thres)
	{
		this->ori_h=raw_img.rows;
    	this->ori_w=raw_img.cols;
		this->sx= 160.0f/this->model_input_w;
        this->sy= 160.0f/this->model_input_h;

		//load the .pt model
		try
		{
			this->module = torch::jit::load(model_path);
            printf("The model load success!\n");
		}
		catch (const c10::Error& e)
		{
			std::cerr << "The model can't load\n";
			
		}
		

	}

    cv::Mat PreprocessImage()
    {
        //lettter_box method
        cv::Mat frame;
        cv::cvtColor(this->raw_img, frame, cv::COLOR_BGR2RGB);
        // new hw = raw hw * (standard hw / raw hw )=ratio

        this-> ratio = std::min(static_cast<float>(this->model_input_h) / frame.rows,
                            static_cast<float>(this->model_input_w) / frame.cols);
        int newh=(int) std::round(frame.rows*this->ratio);
        int neww=(int) std::round(frame.cols*this->ratio);
        cv::Size new_unpad(neww,newh);
        //get the padding length in each size
        float dw=(this->model_input_w-neww)/2;
        float dh=(this->model_input_h-newh)/2;

        if (neww !=this->model_input_w || newh !=this->model_input_h)
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
       


        cv::copyMakeBorder(frame, frame, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return frame;
    }
    

    std::vector<torch::Tensor> MainInference() {
        // 1. Preprocess the image
        cv::Mat image = this->PreprocessImage();

        // 2. Resize the image to (input_shape_h, input_shape_w, 3)
        cv::Mat input;
        cv::resize(image, input, cv::Size(this->model_input_w, this->model_input_h));

        // 3. Convert BGR (OpenCV format) to RGB
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        // 4. Convert OpenCV Mat to torch Tensor
        torch::Tensor tensor_image = torch::from_blob(input.data, { 1, input.rows, input.cols, 3 }, torch::kByte);

        // 5. Reshape the tensor to (batchsize, channels, height, width)
        tensor_image = tensor_image.permute({ 0, 3, 1, 2 }).to(torch::kFloat);

        // 6. Normalize the image
        tensor_image = tensor_image.div(255.0);  // Normalize the image to [0, 1]

        // Set the device (CPU or GPU)
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        try {
            this->module.to(device);
            std::cout << "Model moved to device: " << device << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Model loading to device failed: " << e.what() << std::endl;
        }

        // Inference
        this->module.eval();
        std::vector<torch::Tensor> output_list;

        try {
            // Perform forward pass
            auto result = module.forward({ tensor_image.to(device) });

            // Check if the result is a tuple
            if (result.isTuple()) {
                std::cout << "Output is a tuple. Extracting the elements." << std::endl;

                // Convert IValue to a tuple
                auto tuple = result.toTuple();

                // Ensure that the tuple has the expected number of elements
                if (tuple->elements().size() >= 2) {
                    // Extract the first tensor (e.g., segmentation result)
                    torch::Tensor detect_coeff_output = tuple->elements()[0].toTensor();
                    std::cout << "detect_coeff shape: " << detect_coeff_output.sizes() << std::endl;

                    // Extract the second tensor (e.g., additional output like bounding boxes)
                    torch::Tensor mask_output = tuple->elements()[1].toTensor();
                    std::cout << "mask Output shape: " << mask_output.sizes() << std::endl;

                    // Push to the output list
                    output_list.push_back(detect_coeff_output);
                    output_list.push_back(mask_output);
                } else {
                    std::cerr << "Error: The tuple does not contain enough elements!" << std::endl;
                }
            } else {
                std::cerr << "Error: Expected tuple, but got something else!" << std::endl;
            }

        } catch (const c10::Error& e) {
            std::cerr << "Can't get output! Error: " << e.what() << std::endl;
        }

        return output_list;
    }


    void postprocess(std::vector<torch::Tensor> output_list)
    { 
        torch::Tensor detect_coeff_output = output_list[0];
        detect_coeff_output = detect_coeff_output.squeeze(0).permute({1,0});  // Shape: [116, 8400] --> [8400, 116]
        std::cout << "detect_coeff shape: " << detect_coeff_output.sizes() << std::endl;

        torch::Tensor mask_output = output_list[1];
        mask_output = mask_output.squeeze(0);  // Shape: [32, 160, 160]
        std::cout << "mask Output shape: " << mask_output.sizes() << std::endl;

        // Get mask coefficients from detect coefficients [8400, 32]
        auto mask_coeff_tensor = detect_coeff_output.split({4, this->num_classes, 32}, 1)[2];
        // reshape (32, 160*160)
        torch::Tensor mask_output_reshaped = mask_output.view({32, -1});  // 32x160*160
        // matrix operation (8400, 32) and (32, 160*160)
        torch::Tensor mask_result = torch::mm(mask_coeff_tensor, mask_output_reshaped);  
        std::cout << "Result shape: " << mask_result.sizes() << std::endl;
        mask_result = torch::sigmoid(mask_result);
        cv::Mat mask_result_mat(mask_result.size(0), mask_result.size(1), CV_32F, mask_result.data_ptr<float>());


        
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;
        

        // Get the detection boxes and process them
        for (int i = 0; i < detect_coeff_output.sizes()[0]; i++) {
            auto predict_sample = detect_coeff_output[i];
            auto prediction_split = predict_sample.split({4, this->num_classes, 32}, 0);
            auto box_coord = prediction_split[0];
            auto class_scores = prediction_split[1];

           

            // Find the max confidence class and its index
            auto [class_confidence, class_index] = class_scores.max(0, true);

            // Extract box coordinates in standard model input hw
            int cx = static_cast<int>(box_coord[0].item<float>());
            int cy = static_cast<int>(box_coord[1].item<float>());
            int ow = static_cast<int>(box_coord[2].item<float>());
            int oh = static_cast<int>(box_coord[3].item<float>());
            //(raw hw/standard hw   )= 1/ratio
            cx = (cx - this->left) / this->ratio;
            cy = (cy - this->top) / this->ratio;
            ow = ow / this->ratio;
            oh = oh / this->ratio;

            // Transform box to original image size
            int x = static_cast<int>(cx - 0.5 * ow );
            int y = static_cast<int>(cy - 0.5 * oh );
            int width = static_cast<int>(ow );
            int height = static_cast<int>(oh);

            cv::Rect box(x, y, width, height);
            boxes.push_back(box);
            classIds.push_back(class_index.item<int>());
            confidences.push_back(class_confidence.item<float>());
        }

        // Perform Non-Maximum Suppression (NMS)
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, this->conf_thres, this->iou_thres, indexes);
        std::cout << "index: " << indexes.size() << std::endl;

        // Traverse each box and get the instance segmentation mask
        for (int j = 0; j < indexes.size(); j++) {
            int idx = indexes[j];
            int cid = classIds[idx];
            cv::Rect box_ = boxes[idx];
            this->box_result_list.push_back(boxes[idx]);

            cv::Mat m=mask_result_mat.row(idx);
            // Convert 1x25600 to 1x160x160 (shape adjustment)
            cv::Mat m1 = m.reshape(1, 160);  
            std::cout << m.size() << std::endl;

            // Get the ROI from the mask
            int x1 = std::max(0, box_.x);
            int y1 = std::max(0, box_.y);
            int x2 = std::max(0, box_.br().x);
            int y2 = std::max(0, box_.br().y);

            // Rescale mask coordinates to fit the box dimensions
            // the reverse process of letter-box
            // (x1+this->left)* this->ratio: from image original size to standard model input size (eg:640x640)
            //(x1+this->left)* this->ratio*sx: from standard model input size to feature map size (160x160)
            int mx1 = std::max(0, static_cast<int>(((x1+this->left) * sx) * this->ratio));
            int mx2 = std::max(0, static_cast<int>(((x2+this->left) * sx) * this->ratio));
            int my1 = std::max(0, static_cast<int>(((y1+this->top) * sy)  *this->ratio));
            int my2 = std::max(0, static_cast<int>(((y2+this->top) * sy) * this->ratio));

            // Fix out-of-range box boundaries
            if (mx2 >= m1.cols) {
                mx2 = m1.cols - 1;
            }
            if (my2 >= m1.rows) {
                my2 = m1.rows - 1;
            }

            // Get the ROI inside the box and resize it
            cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
            cv::Mat rm;
            cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

            // Threshold the mask to binary values
            for (int r = 0; r < rm.rows; r++) {
                for (int c = 0; c < rm.cols; c++) {
                    float pv = rm.at<float>(r, c);
                    if (pv > 0.5) {
                        rm.at<float>(r, c) = 1.0;
                    } else {
                        rm.at<float>(r, c) = 0.0;
                    }
                }       
            }

            this->mask_result_list.push_back(rm);
        }
    }

    void Visualize()
    {
        cv::Mat output_img= this->raw_img.clone();
        cv::RNG rng;
        cv::Mat rgb_mask=cv::Mat ::zeros(output_img.size(), output_img.type());
        std::cout<<"1:"<<this->box_result_list.size()<<std::endl;
        std::cout<<"1:"<<this->mask_result_list.size()<<std::endl;
        assert(this->box_result_list.size() == this->mask_result_list.size() && "The sizes of box_result_list and mask_result_list are not equal!");
        for (int i=0;i< this->box_result_list.size();i++ )
        {
            cv::Rect box =this->box_result_list[i];
            int x1 = std::max(0, box.x);
            int y1 = std::max(0, box.y);
            int x2 = std::max(0, box.br().x);
            int y2 = std::max(0, box.br().y);
            cv::rectangle(output_img, box, cv::Scalar(0, 0, 255), 2, 8, 0);
            cv::Mat predict_mask= this->mask_result_list[i];
            //give random colour
            predict_mask=predict_mask*rng.uniform(0,255);
            cv::Mat det_mask;
            predict_mask.convertTo(det_mask,CV_8UC1);

            //FIX outside range
            if ((y1 + det_mask.rows) >= output_img.rows) {
			    y2 = output_img.rows - 1;
            }
            if ((x1 + det_mask.cols) >= output_img.cols) {
                x2 = output_img.cols - 1;
            }
            // blanket image
            cv::Mat mask = cv::Mat::zeros(cv::Size(output_img.cols, output_img.rows), CV_8UC1);
            
            // copy the target mask to blanket mask 
            det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
            //blend rgb_mask and mask
            add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);

        }
        //add original image and mask
        cv::Mat result;
        cv::addWeighted(output_img, 0.5, rgb_mask, 0.5, 0, result);
        result.copyTo(output_img);
        cv::imwrite(this->result_save_path,output_img);
    }


    void main_process()
    {
        std::vector<torch::Tensor> output_list = this->MainInference();
        this->postprocess( output_list);
        this->Visualize();
    }


};


int main()
{
    std::string model_path="/home/kingargroo/cpp/libtorchunet/yolov8m-seg.torchscript";
    cv::Mat raw_img=cv::imread("/home/kingargroo/Downloads/dataset/test4.jpeg");
	std::string result_save_path="/home/kingargroo/cpp/libtorchunet/result2.png";
    int model_input_h=640;
    int model_input_w=640;
	Yolov8segInference inference(model_path,raw_img,model_input_h,model_input_w,result_save_path,80,0.25,0.45);
	inference.main_process();

    return 0;
}

