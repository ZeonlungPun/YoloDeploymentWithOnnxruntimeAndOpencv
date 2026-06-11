#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
using namespace cv;
using namespace std;


// 保持縱橫比的 Letterbox 預處理
cv::Mat letterbox(const cv::Mat& source, cv::Size target_size, cv::Scalar color, float& scale, int& pad_x, int& pad_y) {
    int w = source.cols;
    int h = source.rows;
    
    // 計算縮放比例 (取最小的那一個，確保影像能完全容納在 target_size 內)
    scale = std::min((float)target_size.width / w, (float)target_size.height / h);
    
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    
    // 等比例縮放影像
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // 計算需要填充的邊界 (對稱填充)
    int dw = target_size.width - new_w;
    int dh = target_size.height - new_h;
    
    pad_x = dw / 2;
    pad_y = dh / 2;
    
    int top = pad_y;
    int bottom = dh - top;
    int left = pad_x;
    int right = dw - left;
    
    // 進行邊界填充 (預設使用灰色 114)
    cv::Mat result;
    cv::copyMakeBorder(resized, result, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    cv::imwrite("letter.png",result);
    return result;
}

static void calculate_and_dump(const Ort::Value &v, const std::string &name) {
    // 1. 检查是否为 Tensor
    if (!v.IsTensor()) {
        std::cerr << "[" << name << "] Error: Input Ort::Value is not a tensor." << std::endl;
        return;
    }

    // 2. 获取 Tensor 的类型和形状信息
    auto type_info = v.GetTensorTypeAndShapeInfo();
    auto element_type = type_info.GetElementType();

    // 3. 确保数据类型是 float (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "[" << name << "] Error: Only float tensors are supported in this dump function." << std::endl;
        return;
    }

    // 4. 获取元素数量
    size_t element_count = type_info.GetElementCount();
    if (element_count == 0) {
        std::cout << "[" << name << "] Tensor is empty." << std::endl;
        return;
    }

    // 5. 获取原始数据指针
    const float* data = v.GetTensorData<float>();

    // 6. 计算 Max 和 Mean
    float max_val = -std::numeric_limits<float>::infinity(); 
    double sum_val = 0.0; // 使用 double 防止累加时精度丢失或溢出

    for (size_t i = 0; i < element_count; ++i) {
        float val = data[i];
        if (val > max_val) {
            max_val = val;
        }
        sum_val += val;
    }

    float mean_val = static_cast<float>(sum_val / element_count);

    // 7. 打印结果
    std::cout << "[" << name << "] "
              << "Count: " << element_count << " | "
              << "Mean: " << mean_val << " | "
              << "Max: " << max_val << std::endl;
}


static void dump_dfl0(const Ort::Value &v) {
    calculate_and_dump(v, "DFL0");
}

static void dump_cls0(const Ort::Value &v) {
    calculate_and_dump(v, "CLS0");
}



struct Target {
    float x, y; // left, top
    float width, height;
    float score;
    int cid;    // 0: head, 1: stand, 2: other
};

struct Tensor {
    const float *data;
    int channels;   // 144
    int height;
    int width;
};  

static float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

void softmax(const float* input, float* output, int len) {
    float max_val = *std::max_element(input, input + len);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; ++i) {
        output[i] /= sum;
    }
}


float expectation(const float* prob, int len) {
    float exp_val = 0.0f;
    for (int i = 0; i < len; ++i) {
        exp_val += i * prob[i];
    }
    return exp_val;
}

void dfl_decode(
    std::vector<Target> &targets,
    const Tensor &P_dfl,        // (1, 64, h, w)
    const Tensor &P_cls,        // (1, cls_num, h, w)
    int stride,
    float conf_thresh
) {
    
    int cls_num = P_cls.channels;     
    int DFL_CH_STRIDE = P_dfl.width * P_dfl.height;
    int CLS_CH_STRIDE = P_cls.width * P_cls.height;


    for (int h = 0; h < P_cls.height; ++h) {
        for (int w = 0; w < P_cls.width; ++w) {
            int base = h * P_cls.width + w;

        //get classification score    
            float max_cls_score = std::numeric_limits<float>::lowest();
            int max_cls_id = -1;

            for (int c = 0; c < P_cls.channels; ++c) {
                float score = P_cls.data[c * CLS_CH_STRIDE + base];
                if (score > max_cls_score) {
                    max_cls_score = score;
                    max_cls_id = c;
                }
            }

    
            float score = sigmoid(max_cls_score);
            if (score < conf_thresh) {
                continue;
            }

            int cid = max_cls_id;

            float dfl_buf[64];
            for (int i = 0; i < 64; i++) {
                dfl_buf[i] = P_dfl.data[i * DFL_CH_STRIDE + base];
            }

            // 计算 tleft, ttop, tright, tbottom
            float sm_result[16];    // softmax 结果
            softmax(dfl_buf + 0, sm_result, 16);
            float tleft = expectation(sm_result, 16);
            softmax(dfl_buf + 16, sm_result, 16);
            float ttop = expectation(sm_result, 16);
            softmax(dfl_buf + 32, sm_result, 16);
            float tright = expectation(sm_result, 16);
            softmax(dfl_buf + 48, sm_result, 16);
            float tbottom = expectation(sm_result, 16);

    
            float left = (w - tleft + 0.5) * stride;
            float top = (h - ttop + 0.5) * stride;
            float right = (w + tright + 0.5) * stride;
            float bottom = (h + tbottom + 0.5) * stride;

            targets.push_back({
                left, top, 
                right - left, bottom - top, 
                score,
                cid
            });
        }
    }
}

std::vector<Target> postprocess_yolov5_dfl2(
    const Tensor &P8_dfl, const Tensor &P8_cls,
    const Tensor &P16_dfl, const Tensor &P16_cls,
    const Tensor &P32_dfl, const Tensor &P32_cls,
    int cls_num,
    float conf_thres = 0.3f,
    float iou_thres = 0.45f
) {
    std::vector<Target> targets;
    dfl_decode(targets, P8_dfl, P8_cls, 8, conf_thres);
    dfl_decode(targets, P16_dfl, P16_cls, 16, conf_thres);
    dfl_decode(targets, P32_dfl, P32_cls, 32, conf_thres);

    std::sort(targets.begin(), targets.end(), [](const Target& a, const Target& b) {
        return a.score > b.score;
    });

    printf("[postprocess] %zu candidates before NMS, top-10:\n", targets.size());
    int top_n = std::min(10, (int)targets.size());
    for (int k = 0; k < top_n; k++) {
        printf("  #%d: cls=%-4d score=%.4f  pos=(%.1f,%.1f) size=(%.1f,%.1f)\n",
               k, targets[k].cid, targets[k].score,
               targets[k].x, targets[k].y,
               targets[k].width, targets[k].height);
    }

    // nms，是否需要支持“类间去重”？
    //    比如：一个站立框得分可能较低，与一个 other 框 iou 大于阈值时，应该如何选择？
    //    这里不做类间去重 ...
    std::vector<Target> final;
    for (size_t i = 0; i < targets.size(); ++i) {
        bool keep = true;
        for (const auto& r : final) {
            float xx1 = std::max(targets[i].x, r.x);
            float yy1 = std::max(targets[i].y, r.y);
            float xx2 = std::min(targets[i].x + targets[i].width, r.x + r.width);
            float yy2 = std::min(targets[i].y + targets[i].height, r.y + r.height);
            float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
            float area1 = targets[i].width * targets[i].height;
            float area2 = r.width * r.height;
            float iou = inter / (area1 + area2 - inter + 1e-6f);

            if (iou > iou_thres) {
                // 类间不去重：不同类别的框不做 NMS 抑制
                if (targets[i].cid != r.cid) continue;

                keep = false;
                break;
            }
        }
        if (keep) final.push_back(targets[i]);
    }

    return final;
}

static void print_targets(const std::vector<Target> &targets, const std::vector<std::string> &labels) {
    printf("===== %zu detections =====\n", targets.size());
    for (size_t i = 0; i < targets.size(); i++) {
        const auto &t = targets[i];
        const char *label = (t.cid >= 0 && t.cid < (int)labels.size()) 
                            ? labels[t.cid].c_str() : "?";
        printf("#%zu: cls=%d(%s) pos=(%.1f,%.1f) size=(%.1f,%.1f) → (%.1f,%.1f) score=%.4f\n",
               i, t.cid, label,
               t.x, t.y, t.width, t.height,
               t.x + t.width, t.y + t.height,
               t.score);
    }
}

static void dump_dfl(const std::vector<Ort::Value> &outs, const std::vector<std::string> &labels) {
    dump_dfl0(outs[0]);
    dump_cls0(outs[1]);
    dump_dfl0(outs[2]);
    dump_cls0(outs[3]);
    dump_dfl0(outs[4]);
    dump_cls0(outs[5]);

    const float *p8_dfl = outs[0].GetTensorData<float>();
    const float *p8_cls = outs[1].GetTensorData<float>();
    const float *p16_dfl = outs[2].GetTensorData<float>();
    const float *p16_cls = outs[3].GetTensorData<float>();
    const float *p32_dfl = outs[4].GetTensorData<float>();
    const float *p32_cls = outs[5].GetTensorData<float>();

    std::vector<Target> targets = postprocess_yolov5_dfl2(
        { p8_dfl, 64, 48, 88}, { p8_cls, 5, 48, 88},
        { p16_dfl, 64, 24, 44}, { p16_cls, 5, 24, 44},
        { p32_dfl, 64, 12, 22}, { p32_cls, 5, 12, 22},
        5,
        0.2f,
        0.45f
    );

    print_targets(targets, labels);
}

// 通用推理函数，返回原图坐标系下的 Target 列表
static std::vector<Target> run_model(
    const cv::Mat &frame,
    const std::string &onnx_path,
    float conf_thres,
    float iou_thres)
{
    Ort::SessionOptions session_options;
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    // 获取输入尺寸
    int input_w = 0, input_h = 0;
    std::vector<std::string> input_names;
    for (size_t i = 0; i < numInputNodes; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names.push_back(name.get());
        auto dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        input_w = dims[3];
        input_h = dims[2];
        printf("[%s] input: %lldx%lldx%lldx%lld\n", onnx_path.c_str(),
               (long long)dims[0], (long long)dims[1], (long long)dims[2], (long long)dims[3]);
    }

    // 获取输出名称及形状
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> output_shapes(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(name.get());
        output_shapes[i] = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        printf("[%s] output[%zu]: %s  shape=", onnx_path.c_str(), i, name.get());
        for (auto d : output_shapes[i]) printf("%lld ", (long long)d);
        printf("\n");
    }

    // Letterbox（两模型输入尺寸相同）
    float scale;
    int pad_x, pad_y;
    cv::Mat padded = letterbox(frame, cv::Size(input_w, input_h), cv::Scalar(114, 114, 114), scale, pad_x, pad_y);

    cv::Mat blob = cv::dnn::blobFromImage(padded, 1 / 255.0, cv::Size(input_w, input_h),
                                          cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape{1, 3, input_h, input_w};

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), tpixels, input_shape.data(), input_shape.size());

    std::vector<const char*> out_names;
    for (auto &n : output_names) out_names.push_back(n.c_str());

    std::array<const char*, 1> in_names = {input_names[0].c_str()};

    std::vector<Ort::Value> outputs;
    try {
        outputs = session.Run(Ort::RunOptions{nullptr}, in_names.data(), &input_tensor, 1,
                              out_names.data(), out_names.size());
    } catch (const std::exception &e) {
        std::cerr << "[" << onnx_path << "] Run failed: " << e.what() << std::endl;
        return {};
    }

    std::vector<Target> targets;

    if (numOutputNodes > 1) {
        //: P8_dfl, P8_cls, P16_dfl, P16_cls, P32_dfl, P32_cls
        //  H,W：P_dfl shape=(1,64,H,W)
        int h8 = (int)output_shapes[0][2], w8 = (int)output_shapes[0][3];
        int h16 = (int)output_shapes[2][2], w16 = (int)output_shapes[2][3];
        int h32 = (int)output_shapes[4][2], w32 = (int)output_shapes[4][3];

        const float *p8_dfl = outputs[0].GetTensorData<float>();
        const float *p8_cls = outputs[1].GetTensorData<float>();
        const float *p16_dfl = outputs[2].GetTensorData<float>();
        const float *p16_cls = outputs[3].GetTensorData<float>();
        const float *p32_dfl = outputs[4].GetTensorData<float>();
        const float *p32_cls = outputs[5].GetTensorData<float>();

        int cls_num = (int)output_shapes[1][1];  // P8_cls shape: (1, cls_num, h, w)

        targets = postprocess_yolov5_dfl2(
            {p8_dfl, 64, h8, w8}, {p8_cls, cls_num, h8, w8},
            {p16_dfl, 64, h16, w16}, {p16_cls, cls_num, h16, w16},
            {p32_dfl, 64, h32, w32}, {p32_cls, cls_num, h32, w32},
            cls_num, conf_thres, iou_thres);

        // dfl_decode 输出在 letterbox 坐标系，转换回原图
        for (auto &t : targets) {
            t.x = (t.x - pad_x) / scale;
            t.y = (t.y - pad_y) / scale;
            t.width /= scale;
            t.height /= scale;
        }

    } else {
        // ===== 1 输出（完整模型）：ONNX 内置后处理 =====
        int C = (int)output_shapes[0][1];  // 4 bbox + cls_num
        int N = (int)output_shapes[0][2];  // 候选数
        int cls_num = C - 4;

        const float *pdata = outputs[0].GetTensorData<float>();
        // 诊断：dump ONNX 输出中对应 P8 位置 (40,36) 和 (17,30) 的候选
        // P8 pos (h,w) → ONNX candidate index = h*W + w
        printf("[onnx-1out] dump specific positions:\n");
        int check_positions[][2] = {{40,36}, {17,30}, {40,37}, {41,36}};
        for (auto &cp : check_positions) {
            int h = cp[0], w = cp[1];
            int idx = h * 88 + w;  // P8 W=88
            if (idx >= N) continue;
            printf("  P8(%d,%d)[idx=%d]: cx=%.2f cy=%.2f w=%.2f h=%.2f  cls=",
                   h, w, idx,
                   pdata[0*N + idx], pdata[1*N + idx],
                   pdata[2*N + idx], pdata[3*N + idx]);
            float ms = 0; int mc = -1;
            for (int c = 4; c < C; c++) {
                printf("%.4f%s", pdata[c*N + idx], c < C-1 ? "," : "");
                if (pdata[c*N + idx] > ms) { ms = pdata[c*N + idx]; mc = c-4; }
            }
            // 反推原始 logit
            float raw = ms > 0 && ms < 1 ? log(ms / (1.f - ms)) : 999;
            printf(" max=%.4f(cls=%d) raw≈%.4f\n", ms, mc, raw);
        }

        // 输出形状: (1, C, N) → 每列是一个候选: [cx, cy, w, h, cls_0..cls_{n-1}]
        // 需转置为 N×C 方便按行处理
        std::vector<float> row(C);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> scores;

        for (int i = 0; i < N; i++) {
            // 提取第 i 个候选的 C 个值
            for (int c = 0; c < C; c++) row[c] = pdata[c * N + i];

            // 找最大类别分数（column 4..C-1 经过 sigmoid）
            float max_score = 0;
            int max_cls = 0;
            for (int c = 4; c < C; c++) {
                if (row[c] > max_score) { max_score = row[c]; max_cls = c - 4; }
            }
            if (max_score < 0.01f) continue;

            float cx = row[0], cy = row[1], ow = row[2], oh = row[3];
            // ONNX 输出在 letterbox 空间，转换回原图
            float x = (cx - 0.5f * ow - pad_x) / scale;
            float y = (cy - 0.5f * oh - pad_y) / scale;
            float w = ow / scale;
            float h = oh / scale;

            x = std::max(0.f, x);
            y = std::max(0.f, y);
            w = std::min(w, (float)frame.cols - x);
            h = std::min(h, (float)frame.rows - y);

            boxes.push_back(cv::Rect((int)x, (int)y, (int)w, (int)h));
            class_ids.push_back(max_cls);
            scores.push_back(max_score);
        }

        // OpenCV NMS
        printf("[onnx-1out] %zu candidates before NMS, top-10:\n", boxes.size());
        int top_n = std::min(10, (int)boxes.size());
        for (int k = 0; k < top_n; k++) {
            printf("  #%d: cls=%-4d score=%.4f  pos=(%d,%d) size=(%d,%d)\n",
                   k, class_ids[k], scores[k],
                   boxes[k].x, boxes[k].y,
                   boxes[k].width, boxes[k].height);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, indices);
        for (int idx : indices) {
            targets.push_back({
                (float)boxes[idx].x, (float)boxes[idx].y,
                (float)boxes[idx].width, (float)boxes[idx].height,
                scores[idx], class_ids[idx]
            });
        }
    }

    printf("[%s] %zu targets after NMS\n", onnx_path.c_str(), targets.size());
    return targets;
}

// 比较两组 Target，按 IoU 双向匹配
static void compare_targets(
    const std::vector<Target> &a, const std::string &name_a,
    const std::vector<Target> &b, const std::string &name_b,
    const std::vector<std::string> &labels)
{
    printf("\n========== Comparison: %s vs %s ==========\n", name_a.c_str(), name_b.c_str());
    printf("%-6s %-10s %-10s %-14s %-14s %-8s %-8s %s\n",
           "Match", "IoU", "Δscore", "box_a(xywh)", "box_b(xywh)", "cls_a", "cls_b", "Note");

    std::vector<bool> matched_b(b.size(), false);
    int matched_count = 0;

    for (size_t i = 0; i < a.size(); i++) {
        float best_iou = 0;
        int best_j = -1;

        for (size_t j = 0; j < b.size(); j++) {
            if (matched_b[j]) continue;
            float xx1 = std::max(a[i].x, b[j].x);
            float yy1 = std::max(a[i].y, b[j].y);
            float xx2 = std::min(a[i].x + a[i].width, b[j].x + b[j].width);
            float yy2 = std::min(a[i].y + a[i].height, b[j].y + b[j].height);
            float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
            float area_a = a[i].width * a[i].height;
            float area_b = b[j].width * b[j].height;
            float iou = inter / (area_a + area_b - inter + 1e-6f);
            if (iou > best_iou) { best_iou = iou; best_j = (int)j; }
        }

        const char *label_a = (a[i].cid >= 0 && a[i].cid < (int)labels.size()) ? labels[a[i].cid].c_str() : "?";
        char box_a_str[64], box_b_str[64];
        snprintf(box_a_str, sizeof(box_a_str), "(%.0f,%.0f %.0fx%.0f)", a[i].x, a[i].y, a[i].width, a[i].height);

        if (best_iou > 0.1f && best_j >= 0) {
            matched_b[best_j] = true;
            matched_count++;
            const auto &bb = b[best_j];
            const char *label_b = (bb.cid >= 0 && bb.cid < (int)labels.size()) ? labels[bb.cid].c_str() : "?";
            snprintf(box_b_str, sizeof(box_b_str), "(%.0f,%.0f %.0fx%.0f)", bb.x, bb.y, bb.width, bb.height);
            float score_delta = a[i].score - bb.score;
            const char *note = (a[i].cid == bb.cid) ? "" : "⚠ CLS MISMATCH";
            printf("a#%-4zu IoU=%-8.4f Δ=%-8.4f %-14s %-14s %d(%-5s) %d(%-5s) %s\n",
                   i, best_iou, score_delta, box_a_str, box_b_str,
                   a[i].cid, label_a, bb.cid, label_b, note);
        } else {
            printf("a#%-4zu IoU=%-8s Δ=%-8s %-14s %-14s %d(%-5s) --       UNMATCHED (b)\n",
                   i, "--", "--", box_a_str, "--", a[i].cid, label_a);
        }
    }

    // 打印 b 中未匹配的
    for (size_t j = 0; j < b.size(); j++) {
        if (!matched_b[j]) {
            const char *label_b = (b[j].cid >= 0 && b[j].cid < (int)labels.size()) ? labels[b[j].cid].c_str() : "?";
            printf("b#%-4zu %-10s %-10s %-14s (%.0f,%.0f %.0fx%.0f) %-5s %d(%-5s) UNMATCHED (a)\n",
                   j, "--", "--", "--", b[j].x, b[j].y, b[j].width, b[j].height, "--", b[j].cid, label_b);
        }
    }

    printf("Summary: %zu(a) %zu(b) → %d matched, %zu unmatched(a), %zu unmatched(b)\n",
           a.size(), b.size(), matched_count,
           a.size() - matched_count, b.size() - matched_count);

    // 按 IoU 统计
    if (matched_count > 0) {
        printf("✓ postprocess_yolov5_dfl2 matches ONNX built-in postprocess\n");
    } else {
        printf("✗ NO matches found — postprocess_yolov5_dfl2 is likely WRONG\n");
    }
}

int main(int argc, char** argv) {
    const char *img_path = "build_pc/test.png";
    const char *onnx_full = "build_pc/best.onnx";
    const char *onnx_cut  = "build_pc/best_cut.onnx";

    std::vector<std::string> labels = {"head", "stand", "other", "3", "4"};

    cv::Mat frame = cv::imread(img_path);
    if (frame.empty()) {
        std::cerr << "Failed to load: " << img_path << std::endl;
        return 1;
    }
    printf("load image: %dx%d\n\n", frame.cols, frame.rows);

    float conf_thres = 0.25f;
    float iou_thres  = 0.45f;

    // 1-output 完整模型（ONNX 内置后处理 + OpenCV NMS）
    auto targets_full = run_model(frame, onnx_full, conf_thres, iou_thres);
    printf("\n--- %s ---\n", onnx_full);
    print_targets(targets_full, labels);

    // 6-output 裁剪模型（postprocess_yolov5_dfl2 解码）
    auto targets_cut = run_model(frame, onnx_cut, conf_thres, iou_thres);
    printf("\n--- %s ---\n", onnx_cut);
    print_targets(targets_cut, labels);

    // 对比
    compare_targets(targets_full, onnx_full,
                   targets_cut, onnx_cut,
                   labels);

    return 0;
}