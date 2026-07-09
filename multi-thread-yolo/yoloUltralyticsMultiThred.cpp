/*
 * YolovUltralyticsInference: A YOLOv8-based inference class for object detection with multi-threading support
 * 
 * Author: Zeonlung Pun
 * Date: July 5th, 2026
 * 
 * Description:
 * This class provides an implementation of Ultralytics YOLO series object detection algorithms inference 
 * using ONNX and OpenCV.
 * It includes methods for reading the model, preprocessing input images, running inference,
 * and drawing detection results on the input image.
 * To improve efficiency, this implementation supports multi-threading for concurrent inference on video streams using a producer-consumer model.
 * 
 * In includes three threads: preprocessing, inference, and postprocessing. 
 * Each thread handles a specific stage of the pipeline, allowing for parallel processing of frames.
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
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/times.h>
#include <sys/vfs.h>

// 結構體：用來記錄進程的 CPU 時間片
struct CPU_Time {
    unsigned long utime;  // 使用者態時間
    unsigned long stime;  // 核心態時間
    double real_time;     // 壁鐘時間（真實世界時間）
};

// 獲取當前進程的 CPU 狀態
CPU_Time get_process_cpu_time() {
    CPU_Time t;
    std::ifstream stat_file("/proc/self/stat");
    std::string tmp;
    
    // /proc/self/stat 的第 14 和 15 個欄位分別是 utime 和 stime
    // 我們跳過前 13 個欄位
    for (int i = 0; i < 13; ++i) stat_file >> tmp;
    stat_file >> t.utime >> t.stime;
    
    // 獲取當前高精度真實時間
    auto now = std::chrono::high_resolution_clock::now();
    t.real_time = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() / 1000000.0;
    
    return t;
}

// 計算平均 CPU 佔用率
double calculate_cpu_usage(const CPU_Time& start, const CPU_Time& end) {
    long hertz = sysconf(_SC_CLK_TCK); // 獲取系統每秒的時鐘滴答數 (通常是 100)
    
    unsigned long delta_user = end.utime - start.utime;
    unsigned long delta_sys = end.stime - start.stime;
    double delta_time = end.real_time - start.real_time;
    
    if (delta_time <= 0) return 0.0;
    
    // 總佔用時間片（秒） = (使用者態 + 核心態) / 系統赫茲
    double total_cpu_time_seconds = (double)(delta_user + delta_sys) / hertz;
    
    // CPU 佔用率百分比
    return (total_cpu_time_seconds / delta_time) * 100.0;
}


std::mutex log_mutex;
void safe_log(const std::string& msg) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << msg << std::endl;
}

// a structure to hold the data for each frame in the pipeline
struct FrameData
{
    cv::Mat input_image; // Original input image
    cv::Mat preprocessed_image; // Preprocessed image for inference
    int image_id; // Unique identifier for the frame
    std::vector<cv::Rect> boxes; // Detected bounding boxes
    std::vector<int> classIds; // Class IDs for detected objects
    std::vector<float> confidences; // Confidence scores for detected objects
    cv::Mat blob; // Blob for inference
    int top,left; //to record the padding length in each size
    float ratio; // to record the ratio of the original image to the model input size
    float x_factor = 0;
    float y_factor = 0;

  
};





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
    float confidence_thres;
    float iou_thres;
    int model_input_w, model_input_h, model_output_h, model_output_w;
    float x_factor, y_factor,ratio;
    std::vector<std::string> preprocess_method;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

    YolovUltralyticsInference(
        std::vector<std::string> labels,
        std::string onnx_path_name,
        float confidence_thres = 0.05,
        float iou_thres = 0.3)
        : labels(labels), onnx_path_name(onnx_path_name), 
            confidence_thres(confidence_thres),
          iou_thres(iou_thres), env(ORT_LOGGING_LEVEL_ERROR, "yolov-onnx"), session_options(),
          session(nullptr)
    {
        this->model_input_w = 0;
        this->model_input_h = 0;
        this->model_output_h = 0;
        this->model_output_w = 0;
      

        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        session_options.SetIntraOpNumThreads(2);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
     
    
        read_model();
    }

    void read_model() {
        /*
        Reads the ONNX model file and initializes the inference session.
        Only called once during the initialization of the YolovUltralyticsInference class.
        not need to be protected by mutex, as it is called only once during initialization.
        */
        std::ifstream infile(this->onnx_path_name);
        if (!infile.good()) {
            throw std::runtime_error("ONNX model file not found: " + this->onnx_path_name);
        }

        session = Ort::Session(env, this->onnx_path_name.c_str(), session_options);

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

        
    }


    void preprocess(cv::Mat src_img,FrameData & frameData) {
               /*
	Preprocesses the input image before performing inference.
	belongs to the preprocessing thread.

	*/
        frameData.input_image = src_img.clone();
        // Convert the image color space from BGR to RGB
        cv::cvtColor(frameData.input_image, frameData.preprocessed_image, cv::COLOR_BGR2RGB);

        // Preprocesses the input image according to different selected method
		
        // reference in yolov5
        float ratio = std::min(static_cast<float>(this->model_input_h) / frameData.input_image.rows, 
                    static_cast<float>(this->model_input_w) / frameData.input_image.cols);
        int newh=(int) std::round(frameData.input_image.rows*ratio);
        int neww=(int) std::round(frameData.input_image.cols*ratio);
        cv::Size new_unpad(neww,newh);
        //get the padding length in each size
        float dw=(this->model_input_w-neww)/2;
        float dh=(this->model_input_h-newh)/2;

        if (neww !=this->model_input_w || newh !=this->model_input_h)
        {  //resize the image with same ratio for wdith and height
            cv::resize(frameData.input_image,frameData.preprocessed_image,new_unpad,cv::INTER_LINEAR);
        }
        // calculate the padding pixel around 
        int top,bottom,left,right;
        top =(int) std::round(dh-0.1);
        bottom= (int) std::round(dh+0.1);
        left = (int) std::round(dw-0.1);
        right= (int) std::round(dw+0.1);
        
        frameData.top=top;
        frameData.left=left;
        frameData.ratio=ratio;
        
        cv::copyMakeBorder(frameData.preprocessed_image, frameData.preprocessed_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        cv::dnn::blobFromImage(frameData.input_image,frameData.blob, 1 / 255.0, cv::Size(this->model_input_w,this->model_input_h), cv::Scalar(0, 0, 0), true, false);

        frameData.x_factor = frameData.input_image.cols / static_cast<float>(this->model_input_w);
        frameData.y_factor = frameData.input_image.rows / static_cast<float>(this->model_input_h);
        
		
    }


    void MainInference(FrameData & frameData) {
        /*
        Performs inference using the ONNX model on preprocessed images.
        This function is intended to be run in a separate thread and waits for preprocessed frames to be available.
        It retrieves preprocessed frames, runs inference, and stores the results for post-processing.
        */

         // Preprocess the image
        cv::Mat blob= frameData.blob;

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
            return ;
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

                
                // Adjust coordinates based on padding and scaling
                cx = (cx - frameData.left) / frameData.ratio;
                cy = (cy - frameData.top) / frameData.ratio;
                ow = ow / frameData.ratio;
                oh = oh / frameData.ratio;
                
                int x = static_cast<int>(cx - 0.5 * ow);
                int y = static_cast<int>(cy - 0.5 * oh);
                int width = static_cast<int>(ow);
                int height = static_cast<int>(oh);

               
               
                cv::Rect box(x, y, width, height);
                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, this->confidence_thres, this->iou_thres, indexes);

        for (size_t i = 0; i < indexes.size(); ++i) {
            int idx = indexes[i];
            frameData.boxes.push_back(boxes[idx]);
            frameData.classIds.push_back(classIds[idx]);
            frameData.confidences.push_back(confidences[idx]);
        }



    }

    void draw_detections(FrameData & frameData) {
    /*
	  Draws bounding boxes and labels on the input image based on the detected objects.
	  It belongs to the post-processing thread and is called after inference is completed.
	*/
        for (size_t i = 0; i < frameData.boxes.size(); i++) {
            
            cv::rectangle(frameData.input_image, frameData.boxes[i], cv::Scalar(0, 0, 255), 2, 8);
            std::string label = labels[frameData.classIds[i]];
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(
            label,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
            &baseline);
            const cv::Rect &box = frameData.boxes[i];
            int x = box.x;
            int y = std::max(box.y - 5, textSize.height);
            cv::rectangle(frameData.input_image,
                cv::Point(x, y - textSize.height - baseline),
                cv::Point(x + textSize.width, y + baseline),
                cv::Scalar(0,0,0),
                cv::FILLED);
            cv::putText(frameData.input_image,
              label,
              cv::Point(x, y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6,
              cv::Scalar(255,255,255),
              2);
        }
        
    }
   


};

//first layer: Preprocess -> Inference
std::deque<FrameData> preprocessed_pending; // deque to hold preprocessed frames,FIFO 
std::mutex preprocessed_mutex; // Mutex for synchronizing access to preprocessed data
std::condition_variable cv_pre_notfull,cv_pre_notempty; // Condition variable for signaling when preprocessing is done

//second layer: Inference -> Postprocess
std::deque<FrameData> inference_pending; // deque to hold frames ready for inference,FIFO 
std::mutex inference_mutex; // Mutex for synchronizing access to inference data
std::condition_variable cv_inference_notfull,cv_inference_notempty; // Condition variable for signaling when inference is done      


const int MAX_PENDING_TASK = 8; // Maximum number of pending tasks in the pipeline
bool quit = false;

int main()
{
    std::vector<std::string> labels = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    std::string onnx_path_name = "/home/zonekey/project/multi-thread/yolov8n.onnx";
    std::string video_path = "/home/zonekey/video_analyse/libtea/build_x3/test_video/test1.mp4";
    
    // Create an instance of the YolovUltralyticsInference class
    YolovUltralyticsInference yolo(labels, onnx_path_name, 0.25, 0.4);
    yolo.read_model();

    // Load the video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }   
    // set up the video writer to save the output video with detections
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    cv::VideoWriter writer("./result.mp4", 
                           cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(width, height));
    if(!writer.isOpened())
    {
        std::cout<<"VideoWriter open failed"<<std::endl;
        return -1;
    }

    std::atomic<long long> total_preprocess_us{0};
    std::atomic<long long> total_inference_us{0};
    std::atomic<long long> total_postprocess_us{0};
   
    
    int frame_id = 0;

    CPU_Time cpu_start = get_process_cpu_time();

    // Start the preprocessing thread
    std::thread thread_preprocessing([&]() {
        safe_log("[Preprocess] thread kickstart");
        while (true) {
            auto t_start = std::chrono::high_resolution_clock::now();
            cv::Mat frame;
            if (!cap.read(frame)) {
                safe_log("[Preprocess] 影片讀取完畢，正在發送退出訊號...");
                //frames are all read,kickstart the exit system
                std::unique_lock<std::mutex> lock1(preprocessed_mutex);
                std::unique_lock<std::mutex> lock2(inference_mutex);

                quit = true;
                //notify all the threads
                cv_pre_notempty.notify_all();
                cv_pre_notfull.notify_all();
                cv_inference_notempty.notify_all();
                cv_inference_notfull.notify_all();
                break;; // Exit the loop if no more frames are available
            }
             // create a FrameData object to hold the data for this frame
            FrameData frameData;
            frameData.image_id = frame_id++;
            yolo.preprocess(frame, frameData);

            auto t_end = std::chrono::high_resolution_clock::now();
            total_preprocess_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            
            // mutex for the preprocessed_pending
            std::unique_lock<std::mutex> lock_pre(preprocessed_mutex);
            // if here is congestion, it means that the downstream processing is slower than the upstream processing, 
            //we need to wait for the downstream to catch up
            cv_pre_notfull.wait(lock_pre, []
                { return preprocessed_pending.size() < MAX_PENDING_TASK || quit; });
            if (quit) return;
            // 每 30 幀打印一次進度，避免日誌刷屏
            if (frameData.image_id % 30 == 0 || total_frames - frameData.image_id < 5) {
                safe_log("[Preprocess] 已預處理 Frame ID: " + std::to_string(frameData.image_id));
            }

            // Add the preprocessed frame to the queue
            // and notify the inference thread that a new frame is available for processing
            preprocessed_pending.push_back(frameData);
            //notify the inference thread that a new frame is available for processing
            cv_pre_notempty.notify_one();
        }

        std::cout << "Preprocessing thread ended.\n";
    });     

    // Start the inference thread
    std::thread thread_inference([&]() {
        while (true) 
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            //get data from the preprocessed_pending queue
            FrameData frameData;
            {
                std::unique_lock<std::mutex> lock(preprocessed_mutex);
                
                cv_pre_notempty.wait(lock, []
                    { return !preprocessed_pending.empty()|| quit; });
                if (preprocessed_pending.empty() && quit) break;
                if (quit ) return;

                frameData = preprocessed_pending.front();
                preprocessed_pending.pop_front();

                //notify the preprocessing thread that there is space in the queue
                cv_pre_notfull.notify_one();
            }

            //get the end signal from the preprocessing thread, if the preprocessing thread has ended and the preprocessed_pending queue is empty, then we can end the inference thread
            if (frameData.input_image.empty()) {
                
                return;
            }
            
            // implement the inference on the preprocessed frame
            yolo.MainInference(frameData);
            auto t_end = std::chrono::high_resolution_clock::now();
            total_inference_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            // finish inference, push the frameData to the inference_pending queue for post-processing
            {
                std::unique_lock<std::mutex> lock_infer(inference_mutex);
                cv_inference_notfull.wait(lock_infer, []
                    { return inference_pending.size() < MAX_PENDING_TASK || quit; });
                
                if(quit && inference_pending.empty()) break;    
                inference_pending.push_back(frameData);
                //notify the post-processing thread that a new frame is available for processing
                cv_inference_notempty.notify_one();
            }
                
            
        }
        std::cout << "Inference thread ended."<<std::endl;
    });


    
    // Start the post-processing thread
    std::thread thread_postprocessing([&]() {
        while (true) {
            auto t_start = std::chrono::high_resolution_clock::now();
            // get data from the inference_pending queue
            FrameData frameData;
            {
                std::unique_lock<std::mutex> lock_post(inference_mutex);
                cv_inference_notempty.wait(lock_post, []
                    { return !inference_pending.empty() || quit; });
                if (inference_pending.empty() && quit) break;
                
                frameData = inference_pending.front();
                inference_pending.pop_front();

                //notify the inference thread that there is space in the queue
                cv_inference_notfull.notify_one();
            }
            
            yolo.draw_detections(frameData);
            // Save the result image to the output video
            writer.write(frameData.input_image);
            auto t_end = std::chrono::high_resolution_clock::now();
            total_postprocess_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
           
   
            
                        
        }
        std::cout << "Post-processing thread ended."<<std::endl;
    });
        
    auto start_time = std::chrono::high_resolution_clock::now();

    thread_preprocessing.join();
    thread_inference.join();
    thread_postprocessing.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;

    CPU_Time cpu_end = get_process_cpu_time();
    double avg_cpu_usage = calculate_cpu_usage(cpu_start, cpu_end);
    int total_cores = sysconf(_SC_NPROCESSORS_ONLN); // 獲取機器總核心數

    double total_seconds = total_duration.count();
    double FPS =  total_frames / total_seconds;
    double avg_preprocess = (total_preprocess_us / total_frames) / 1000.0;
    double avg_inference  = (total_inference_us / total_frames) / 1000.0;
    double avg_postprocess = (total_postprocess_us / total_frames) / 1000.0;
   

    std::cout<< "============= multi-thread evaluation============="<<std::endl;
    std::cout<< "multi-thread FPS:" << FPS << std::endl;
    std::cout << "average preprocess time:"<<avg_preprocess<<std::endl;
    std::cout << "average inference time:"<<avg_inference<<std::endl;
    std::cout << "average postprocess time:"<<avg_postprocess<<std::endl;
    std::cout << "Actual Total Run Time : " << total_seconds << " seconds" << std::endl;
    std::cout << "  - process avg CPU usage: " << avg_cpu_usage << " %\n";
    std::cout << "  - system usable CPU : " << total_cores << " cores\n";
    std::cout << "  - consuming CPU : " << (avg_cpu_usage / 100.0) << "cores\n";
    std::cout << "=======================================" << std::endl;
    
   
    writer.release();
    std::cout << "All pipeline finished successfully." << std::endl;

    // make comparision with single thread
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::VideoWriter writer_single("./result2.mp4", 
        cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(width, height));
    
    std::atomic<long long> total_preprocess_single{0};
    std::atomic<long long> total_inference_single{0};
    std::atomic<long long> total_postprocess_single{0};
    int frame_id_s = 0;
    
    CPU_Time cpu_start_single = get_process_cpu_time();
    auto start_time_single = std::chrono::high_resolution_clock::now();
    while (true)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if(!cap.read(frame))
        {
            break;
        }
        FrameData frameData;
        frameData.image_id = frame_id_s++;
        yolo.preprocess(frame, frameData);
        auto t_end_pre = std::chrono::high_resolution_clock::now();
        total_preprocess_single += std::chrono::duration_cast<std::chrono::microseconds>(t_end_pre - t_start).count();
        yolo.MainInference(frameData);
        auto t_end_infer = std::chrono::high_resolution_clock::now();
        total_inference_single += std::chrono::duration_cast<std::chrono::microseconds>(t_end_infer - t_end_pre).count();
        yolo.draw_detections(frameData);
        writer_single.write(frameData.input_image);
        auto t_end_post = std::chrono::high_resolution_clock::now();
        total_postprocess_single += std::chrono::duration_cast<std::chrono::microseconds>(t_end_post - t_end_infer).count();
        

    }
    auto end_time_single = std::chrono::high_resolution_clock::now();
    CPU_Time cpu_end_single = get_process_cpu_time();
    double avg_cpu_usage_single = calculate_cpu_usage(cpu_start_single, cpu_end_single);
    

    std::chrono::duration<double> total_duration_single = end_time_single - start_time_single;
    double total_seconds_single = total_duration_single.count();
    double FPS_single =  total_frames / total_seconds_single;
    double avg_preprocess_single = (total_preprocess_single / total_frames) / 1000.0;
    double avg_inference_single  = (total_inference_single / total_frames) / 1000.0;
    double avg_postprocess_single = (total_postprocess_single / total_frames) / 1000.0;
   

    std::cout<< "============= single-thread evaluation============="<<std::endl;
    std::cout<< "single-thread FPS:" << FPS_single << std::endl;
    std::cout << "average preprocess time:"<<avg_preprocess_single<<std::endl;
    std::cout << "average inference time:"<<avg_inference_single<<std::endl;
    std::cout << "average postprocess time:"<<avg_postprocess_single<<std::endl;
    std::cout << "Actual Total Run Time : " << total_seconds_single << " seconds" << std::endl; 
    std::cout << "  - process avg CPU usage: " << avg_cpu_usage_single << " %\n";
    std::cout << "  - system usable CPU : " << total_cores << " cores\n";
    std::cout << "  - consuming CPU : " << (avg_cpu_usage_single / 100.0) << "cores\n";
    std::cout << "=======================================" << std::endl;
    cap.release();
    writer_single.release();
    



    return 0;


}