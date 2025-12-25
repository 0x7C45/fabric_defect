# 在你的PC上运行这个脚本，导出模型
from ultralytics import YOLO
import torch

# 加载你训练好的模型
model = YOLO('train/weights/best.pt')

# 导出为ONNX格式
model.export(format='onnx', imgsz=640, simplify=True)
