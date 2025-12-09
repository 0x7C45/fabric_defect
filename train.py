import os
from ultralytics import YOLO

def train_yolov8():
    """在当前目录下训练YOLOv8模型"""

    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 配置训练参数
    config = {
        'data': os.path.join(current_dir, 'yolo_dataset', 'data.yaml'),
        'model': 'yolov8n.pt',  # 可以使用 yolov8s.pt, yolov8m.pt 等不同大小的模型
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': '0',  # GPU设备，如果是CPU使用 'cpu'
        'project': current_dir,  # 项目目录设为当前目录
        'name': 'train',  # 训练结果文件夹名称
        'exist_ok': True,  # 如果文件夹存在，则继续使用
        'save': True,  # 保存训练检查点和结果
        'save_period': 10,  # 每10个epoch保存一次
        'workers': 4,  # 数据加载的进程数
        'patience': 50,  # 早停的耐心值
        'verbose': True,  # 显示详细输出
        'cos_lr': True,  # 使用余弦学习率调度
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.01,  # 最终学习率因子
        'momentum': 0.937,  # 动量
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3.0,  # 热身epochs
        'warmup_momentum': 0.8,  # 热身动量
        'warmup_bias_lr': 0.1,  # 热身偏置学习率
    }

    # 检查数据文件是否存在
    if not os.path.exists(config['data']):
        print(f"错误: 未找到数据配置文件 {config['data']}")
        print("请先运行数据转换脚本创建yolo_dataset文件夹")
        return

    print("=" * 60)
    print("开始YOLOv8训练")
    print("=" * 60)
    print(f"工作目录: {current_dir}")
    print(f"数据配置: {config['data']}")
    print(f"模型: {config['model']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)

    # 加载模型
    model = YOLO(config['model'])

    # 开始训练
    results = model.train(**config)

    print("=" * 60)
    print("训练完成！")
    print(f"最佳模型保存在: {os.path.join(current_dir, 'train', 'weights', 'best.pt')}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    train_yolov8()
