from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
import sys

if len(sys.argv) != 3:
    print("Usage: python3 {} <model> <out>".format(sys.argv[0]))
    exit()

print(f"--> Preprocessing {sys.argv[1]} ..")
preprocessed_fname = "preprocessed.onnx"
try:
    quant_pre_process(
        sys.argv[1],
        preprocessed_fname,
        skip_symbolic_shape_inference=True,
    )
except Exception as e:
    print(e)
    preprocessed_fname = sys.argv[1]

print(f"--> Quantizing {sys.argv[2]} ..")
quantize_dynamic(
    model_input=preprocessed_fname,
    model_output=sys.argv[2],
    weight_type=QuantType.QUInt8,
)

import os
if os.path.exists("preprocessed.onnx"):
    print("--> Cleaning up preprocessed model ..")
    os.unlink("preprocessed.onnx")

print("Done.")