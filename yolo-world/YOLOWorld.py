import time
import cv2
import numpy as np
import onnxruntime
class DetectionDrawer():

    def __init__(self, class_names):
        self.class_names = class_names
        self.rng = np.random.default_rng(3)
        self.colors = self.get_colors(len(class_names))

    def __call__(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        return self.draw_detections(image, boxes, scores, class_ids, mask_alpha)

    def update_class_names(self, class_names):
        for i, class_name in enumerate(class_names):
            if class_name not in self.class_names:
                self.colors[i] = self.get_colors(1)
            else:
                self.colors[i] = self.colors[list(self.class_names).index(class_name)]

        self.class_names = class_names

    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):

        if class_ids.shape[0] == 0:
            return image
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        classes = self.class_names[class_ids]
        colors = self.colors[class_ids]

        # Draw bounding boxes and labels of detections
        for box, score, label, color in zip(boxes, scores, classes, colors):
            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            caption = f'{label} {score*100:.2f}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    def get_colors(self, num_classes):
        return self.rng.uniform(0, 255, size=(num_classes, 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou
def read_class_embeddings(embed_path):
    data = np.load(embed_path)
    return data["class_embeddings"], data["class_list"]


class YOLOWorld:

    def __init__(self, path, conf_thres=0.3, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image, class_embeddings):
        return self.detect_objects(image, class_embeddings)

    def detect_objects(self, image, class_embeddings):

        if class_embeddings.shape[1] != self.num_classes:
            raise ValueError(f"Number of classes in the class embeddings should be {self.num_classes}")

        input_tensor = self.prepare_input(image)

        # Perform yoloworld on the image
        outputs = self.inference(input_tensor, class_embeddings)

        return self.process_output(outputs)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor, class_embeddings):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: input_tensor, self.input_names[1]: class_embeddings})

        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = YOLOWorld.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.num_classes = model_inputs[1].shape[1]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':

    model_path = "./yolov8s-worldv2.onnx"
    embed_path = "./class_embeddings.npz"

    # Load class embeddings
    class_embeddings, class_list = read_class_embeddings(embed_path)

    # Initialize YOLO-World object detector
    yoloworld_detector = YOLOWorld(model_path, conf_thres=0.3, iou_thres=0.5)

    # Initialize DetectionDrawer
    drawer = DetectionDrawer(class_list)

    img_url = "/home/punzeonlung/CPP/yolo-world/test2.jpg"
    img = cv2.imread(img_url)

    # Detect Objects
    boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

    # Draw detections
    combined_img = drawer(img, boxes, scores, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
