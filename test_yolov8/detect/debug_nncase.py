#!/usr/bin/env python3
"""
调试nncase环境问题
"""

import os
import sys
import traceback

# 设置nncase插件路径
nncase_path = r"C:\Users\Administrator\AppData\Roaming\Python\Python313\site-packages\nncase"
os.environ['NNCASE_PLUGIN_PATH'] = nncase_path

print("=" * 60)
print("调试nncase环境")
print("=" * 60)

print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")
print(f"NNCASE_PLUGIN_PATH: {os.environ.get('NNCASE_PLUGIN_PATH')}")

# 检查nncase目录是否存在
if os.path.exists(nncase_path):
    print(f"nncase目录存在: {nncase_path}")
    # 列出一些关键文件
    import glob
    dll_files = glob.glob(os.path.join(nncase_path, "*.dll"))
    print(f"找到 {len(dll_files)} 个DLL文件")
    for dll in dll_files[:5]:  # 只显示前5个
        print(f"  {os.path.basename(dll)}")
else:
    print(f"警告: nncase目录不存在: {nncase_path}")

# 尝试导入nncase
print("\n尝试导入nncase...")
try:
    import nncase
    print("[OK] nncase导入成功")
    print(f"nncase版本属性: {hasattr(nncase, '__version__')}")
    if hasattr(nncase, '__version__'):
        print(f"nncase版本: {nncase.__version__}")

    # 检查Simulator类是否存在
    print(f"Simulator类存在: {hasattr(nncase, 'Simulator')}")

    # 尝试创建Simulator实例
    print("\n尝试创建Simulator实例...")
    try:
        sim = nncase.Simulator()
        print("[OK] Simulator实例创建成功")

        # 检查Simulator的方法
        print("\nSimulator可用方法:")
        methods = [m for m in dir(sim) if not m.startswith('_')]
        for method in methods[:10]:  # 只显示前10个
            print(f"  {method}")

    except Exception as e:
        print(f"[ERROR] Simulator创建失败: {e}")
        traceback.print_exc()

except ImportError as e:
    print(f"[ERROR] nncase导入失败: {e}")
    traceback.print_exc()

# 检查kmodel文件
print("\n" + "=" * 60)
print("检查kmodel文件")
print("=" * 60)

kmodel_path = '../../train/weights/best.kmodel'
if os.path.exists(kmodel_path):
    file_size = os.path.getsize(kmodel_path)
    print(f"[OK] kmodel文件存在: {kmodel_path}")
    print(f"文件大小: {file_size} 字节 ({file_size/1024/1024:.2f} MB)")

    # 尝试读取文件
    try:
        with open(kmodel_path, 'rb') as f:
            kmodel_data = f.read(100)  # 只读前100字节
        print(f"文件前100字节: {kmodel_data[:50]}...")
    except Exception as e:
        print(f"[ERROR] 文件读取失败: {e}")
else:
    print(f"[ERROR] kmodel文件不存在: {kmodel_path}")
    # 列出目录内容
    parent_dir = os.path.dirname(kmodel_path)
    if os.path.exists(parent_dir):
        print(f"目录 {parent_dir} 内容:")
        import glob
        for file in glob.glob(os.path.join(parent_dir, "*")):
            print(f"  {os.path.basename(file)}")

# 测试模型加载
print("\n" + "=" * 60)
print("测试模型加载")
print("=" * 60)

try:
    import nncase

    # 创建Simulator
    sim = nncase.Simulator()
    print("[OK] Simulator创建成功")

    # 读取kmodel
    with open(kmodel_path, 'rb') as f:
        kmodel_data = f.read()
    print(f"[OK] kmodel文件读取成功: {len(kmodel_data)} 字节")

    # 尝试加载模型
    print("尝试加载模型...")
    sim.load_model(kmodel_data)
    print("[OK] 模型加载成功!")

    # 检查输入输出
    print(f"\n模型输入数量: {sim.inputs_size}")
    for i in range(sim.inputs_size):
        desc = sim.get_input_desc(i)
        print(f"  输入{i}: dtype={desc.dtype}, shape={desc.shape}")

    print(f"模型输出数量: {sim.outputs_size}")
    for i in range(sim.outputs_size):
        desc = sim.get_output_desc(i)
        print(f"  输出{i}: dtype={desc.dtype}, shape={desc.shape}")

except Exception as e:
    print(f"[ERROR] 模型加载失败: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()

    # 提供调试建议
    print("\n" + "=" * 60)
    print("调试建议:")
    print("1. 确认NNCASE_PLUGIN_PATH指向正确的nncase目录")
    print("2. 检查是否缺少DLL依赖（使用Dependency Walker工具）")
    print("3. 尝试重新安装nncase: pip install nncase --upgrade")
    print("4. 检查kmodel文件是否损坏")

print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)