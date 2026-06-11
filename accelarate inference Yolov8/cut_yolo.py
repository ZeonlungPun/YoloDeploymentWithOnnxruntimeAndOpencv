''' hb_mapper 转换为 x3 模型时，发现

...
...
/model.23/cv3/conv/Conv                             BPU  id(0)     HzSQuantizedConv
/model.23/cv3/act/Mul                               BPU  id(0)     HzLut
/model.24/cv2.2/cv2.2.0/conv/Conv                   BPU  id(0)     HzSQuantizedConv
/model.24/cv3.2/cv3.2.0/conv/Conv                   BPU  id(0)     HzSQuantizedConv
/model.24/cv2.2/cv2.2.0/act/Mul                     BPU  id(0)     HzLut
/model.24/cv3.2/cv3.2.0/act/Mul                     BPU  id(0)     HzLut
/model.24/cv2.2/cv2.2.1/conv/Conv                   BPU  id(0)     HzSQuantizedConv
/model.24/cv3.2/cv3.2.1/conv/Conv                   BPU  id(0)     HzSQuantizedConv
/model.24/cv2.2/cv2.2.1/act/Mul                     BPU  id(0)     HzLut
/model.24/cv3.2/cv3.2.1/act/Mul                     BPU  id(0)     HzLut
/model.24/cv2.2/cv2.2.2/Conv                        BPU  id(0)     HzSQuantizedConv
/model.24/cv3.2/cv3.2.2/Conv                        BPU  id(0)     HzSQuantizedConv
/model.24/Concat_2                                  BPU  id(0)     Concat
/model.24/Reshape_2                                 CPU  --        Reshape
/model.24/Concat_3                                  CPU  --        Concat
/model.24/Split                                     CPU  --        Split
/model.24/dfl/Reshape                               CPU  --        Reshape
/model.24/Sigmoid                                   CPU  --        Sigmoid
/model.24/dfl/Transpose                             CPU  --        Transpose
/model.24/dfl/Softmax                               CPU  --        Softmax
/model.24/dfl/Transpose_1                           CPU  --        Transpose
/model.24/dfl/conv/Conv                             BPU  id(1)     HzSQuantizedConv
/model.24/dfl/Reshape_1                             CPU  --        Reshape
/model.24/Slice                                     CPU  --        Slice
/model.24/Slice_1                                   CPU  --        Slice
/model.24/Sub                                       CPU  --        Sub
/model.24/Add_1                                     CPU  --        Add
/model.24/Add_2                                     CPU  --        Add
/model.24/Sub_1                                     CPU  --        Sub
/model.24/Div_1                                     CPU  --        Div
/model.24/Concat_4                                  CPU  --        Concat
/model.24/Mul_2                                     CPU  --        Mul
/model.24/Concat_5                                  CPU  --        Concat

应该选择 /model.23/Concat, /model.23/Concat_1, /model.23/Concat_2 作为三个输出，后面的统统使用 cpu 在后处理中实现

'''

import onnx
import onnx_graphsurgeon as gs
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--onnx", type=str, default="yolov5nu.onnx")
ap.add_argument("--out", type=str, default="yolov5nu_cut.onnx")
opt = ap.parse_args()

model = onnx.load(opt.onnx)

want_outputs = [
    "/model.24/cv2.0/cv2.0.2/Conv", "/model.24/cv3.0/cv3.0.2/Conv", 
    "/model.24/cv2.1/cv2.1.2/Conv", "/model.24/cv3.1/cv3.1.2/Conv",
    "/model.24/cv2.2/cv2.2.2/Conv", "/model.24/cv3.2/cv3.2.2/Conv",
]

graph = gs.import_onnx(model)
target_outputs = []
for node in graph.nodes:
    if node.name in want_outputs:
        target_outputs.extend(node.outputs)

if not target_outputs:
    print("NOT found outputs ...")
else:
    graph.outputs = target_outputs
    xm = gs.export_onnx(graph)
    xm.ir_version = 6
    onnx.save(xm, opt.out)
    print(f"target_outputs:{target_outputs}")