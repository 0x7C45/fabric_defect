import json
import os
import cv2
import shutil
import random
from pathlib import Path
import yaml

def create_yolo_structure(base_path="yolo_dataset"):
    """创建YOLOv8所需的目录结构"""
    dirs = [
        f"{base_path}/images/train",
        f"{base_path}/images/val",
        f"{base_path}/labels/train",
        f"{base_path}/labels/val",
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    return base_path

def load_and_analyze_annotations(json_path):
    """加载并分析标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 统计缺陷类别
    defect_categories = {}
    image_annotations = {}

    for ann in annotations:
        image_name = ann["name"]
        defect_name = ann["defect_name"]
        bbox = ann["bbox"]

        # 统计缺陷类别
        if defect_name not in defect_categories:
            defect_categories[defect_name] = len(defect_categories)

        # 按图像分组标注
        if image_name not in image_annotations:
            image_annotations[image_name] = []

        image_annotations[image_name].append({
            "defect_name": defect_name,
            "bbox": bbox
        })

    return image_annotations, defect_categories

def find_image_paths(data_train_path, image_name):
    """在defect_Images和normal_Images中查找图像文件"""
    possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 先在defect_Images中查找
    defect_path = os.path.join(data_train_path, "defect_Images", image_name)
    if os.path.exists(defect_path):
        return defect_path

    # 尝试不同的扩展名
    for ext in possible_extensions:
        if image_name.endswith(ext):
            # 已经检查过完整文件名，继续
            continue

        test_path = os.path.join(data_train_path, "defect_Images", image_name.split('.')[0] + ext)
        if os.path.exists(test_path):
            return test_path

    # 然后在normal_Images中查找
    normal_path = os.path.join(data_train_path, "normal_Images", image_name)
    if os.path.exists(normal_path):
        return normal_path

    # 尝试不同的扩展名
    for ext in possible_extensions:
        if image_name.endswith(ext):
            continue

        test_path = os.path.join(data_train_path, "normal_Images", image_name.split('.')[0] + ext)
        if os.path.exists(test_path):
            return test_path

    return None

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """将[x_min, y_min, x_max, y_max]转换为YOLO格式"""
    x_min, y_min, x_max, y_max = bbox

    # 确保坐标在图像范围内
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height - 1))

    # 计算中心点和宽高
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # 归一化
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # 确保值在0-1范围内
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm

def process_dataset(data_train_path, json_path, output_base="yolo_dataset", val_split=0.2):
    """处理整个数据集"""

    # 创建YOLO目录结构
    yolo_base = create_yolo_structure(output_base)

    # 加载标注数据
    print("正在加载标注文件...")
    image_annotations, defect_categories = load_and_analyze_annotations(json_path)

    print(f"找到 {len(defect_categories)} 种缺陷类别:")
    for defect, idx in defect_categories.items():
        print(f"  {idx}: {defect}")

    # 获取所有图像名称并随机打乱
    all_images = list(image_annotations.keys())
    random.shuffle(all_images)

    # 划分训练集和验证集
    split_idx = int(len(all_images) * (1 - val_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"\n数据集统计:")
    print(f"  总图像数: {len(all_images)}")
    print(f"  训练集: {len(train_images)}")
    print(f"  验证集: {len(val_images)}")

    # 处理训练集
    train_stats = process_image_set(
        train_images, image_annotations, defect_categories,
        data_train_path, yolo_base, "train"
    )

    # 处理验证集
    val_stats = process_image_set(
        val_images, image_annotations, defect_categories,
        data_train_path, yolo_base, "val"
    )

    # 创建data.yaml配置文件
    create_yaml_config(yolo_base, defect_categories, data_train_path)

    print("\n处理完成!")
    print(f"训练集: {train_stats['processed']} 张图像, {train_stats['annotations']} 个标注")
    print(f"验证集: {val_stats['processed']} 张图像, {val_stats['annotations']} 个标注")
    print(f"\n输出目录: {yolo_base}")
    print(f"配置文件: {yolo_base}/data.yaml")

def process_image_set(image_list, image_annotations, defect_categories,
                     data_train_path, yolo_base, set_type):
    """处理指定集合的图像"""
    stats = {"processed": 0, "annotations": 0, "missing": 0}

    for image_name in image_list:
        # 查找图像文件
        image_path = find_image_paths(data_train_path, image_name)

        if not image_path:
            print(f"警告: 未找到图像 {image_name}")
            stats["missing"] += 1
            continue

        try:
            # 读取图像获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue

            img_height, img_width = img.shape[:2]

            # 复制图像到YOLO目录
            output_image_path = os.path.join(yolo_base, "images", set_type, image_name)
            shutil.copy2(image_path, output_image_path)

            # 创建标签文件
            label_file_path = os.path.join(yolo_base, "labels", set_type,
                                          os.path.splitext(image_name)[0] + ".txt")

            with open(label_file_path, 'w', encoding='utf-8') as f:
                for ann in image_annotations[image_name]:
                    defect_name = ann["defect_name"]
                    bbox = ann["bbox"]

                    # 转换为YOLO格式
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )

                    # 获取类别ID
                    class_id = defect_categories[defect_name]

                    # 写入标签文件
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    stats["annotations"] += 1

            stats["processed"] += 1

            if stats["processed"] % 100 == 0:
                print(f"已处理 {set_type} 集 {stats['processed']} 张图像...")

        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
            continue

    return stats

def create_yaml_config(yolo_base, defect_categories, data_train_path):
    """创建YOLOv8配置文件"""
    # 按ID排序获取类别名称列表
    sorted_defects = sorted(defect_categories.items(), key=lambda x: x[1])
    names = [defect for defect, idx in sorted_defects]

    config = {
        "path": os.path.abspath(yolo_base),
        "train": "images/train",
        "val": "images/val",
        "nc": len(defect_categories),
        "names": names
    }

    yaml_path = os.path.join(yolo_base, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n已创建配置文件: {yaml_path}")

def main():
    """主函数"""
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_train_path = os.path.join(current_dir, "data_train")
    json_path = os.path.join(data_train_path, "Annotations", "anno_train.json")
    output_base = "yolo_dataset"

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 未找到标注文件 {json_path}")
        return

    if not os.path.exists(data_train_path):
        print(f"错误: 未找到数据目录 {data_train_path}")
        return

    print("=" * 60)
    print("织物检测数据集转换工具")
    print("=" * 60)
    print(f"数据目录: {data_train_path}")
    print(f"标注文件: {json_path}")
    print(f"输出目录: {output_base}")
    print("=" * 60)

    # 处理数据集
    try:
        process_dataset(
            data_train_path=data_train_path,
            json_path=json_path,
            output_base=output_base,
            val_split=0.2  # 20%作为验证集
        )

        print("\n" + "=" * 60)
        print("转换完成！现在可以使用YOLOv8训练模型了。")
        print("\n训练命令示例:")
        print(f"yolo detect train data={output_base}/data.yaml model=yolov8n.pt epochs=100 imgsz=640")

    except Exception as e:
        print(f"\n转换过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
