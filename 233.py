# pt模型转onnx
import torch
import onnx
import sys

# 导入yolov5所在的路径，模型转换需要yolov5工程定义的网络结构（即models、utils两个文件夹）
sys.path.insert(0, 'Yolov5/')

model_path = "Yolov5/weights/all_1.pt"
onnx_path = "best.onnx"
model = torch.load(model_path, map_location='cpu')
model = (model.get('ema') or model['model']).float()
if hasattr(model, 'fuse'):
    model.fuse().eval()
else:
    model.eval()

# 设置export为true，裁剪不必要的输出
for k, m in model.named_modules():
    if str(type(m)) == "<class 'models.yolo.Segment'>":
        m.export = True

# Input to the model
x = torch.randn(1, 3, 640, 640, requires_grad=True)
# Export the model
torch.onnx.export(model,  # model being run
                  x,  # model input
                  onnx_path,
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],  # the model's input names
                  output_names=['output0', 'output1'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                                'output0': {0: 'batch', 1: 'anchors'},
                                'output1': {0: 'batch', 2: 'mask_height', 3: 'mask_width'}}
                  )

# Checks
model_onnx = onnx.load(onnx_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Metadata
d = {'stride': int(max(model.stride)), 'names': model.names}
for k, v in d.items():
    meta = model_onnx.metadata_props.add()
    meta.key, meta.value = k, str(v)
onnx.save(model_onnx, onnx_path)
print('process over !!!')