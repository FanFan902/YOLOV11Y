import os
from pathlib import Path
import sys
sys.path.insert(0, '...')
from ultralytics import YOLO

# 设置工作目录
work_dir = Path('...')

# 创建测试结果目录
results_dir = work_dir / 'yolov11Y_test_results'
results_dir.mkdir(exist_ok=True)

# 1. 在验证集上评估模型
print("=== 1. 在验证集上评估模型 ===")

# 加载模型
model_path = 'best.py'
print(f"加载模型: {model_path}")

model = YOLO(model_path)

# 在验证集上评估模型
print(f"\n开始验证...")
metrics = model.val(
    data=work_dir / 'Datas' / 'dataset.yaml',
    device='0',
    imgsz=640,
    batch=16,
    name='val_results',
    project=str(results_dir)
)

# 打印评估指标
print(f"\n评估结果：")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"精确率: {metrics.box.mp:.4f}")
print(f"召回率: {metrics.box.mr:.4f}")

# 2. 对测试集进行预测
print("\n=== 2. 对测试集进行预测 ===")

test_images_dir = work_dir / 'Datas' / 'test' / 'images'
if not test_images_dir.exists():
    print(f"❌ 测试集目录不存在: {test_images_dir}")
else:
    image_count = len(list(test_images_dir.glob('*')))
    print(f"测试集图片数量: {image_count}")

    if image_count > 0:
        results = model.predict(
            source=str(test_images_dir),
            device='0',
            imgsz=640,
            conf=0.25,
            save=True,
            save_txt=True,
            save_conf=True,
            show=False,
            name='test_predictions',
            project=str(results_dir)
        )
        print(f"✅ 预测完成，结果保存在: {results_dir / 'test_predictions'}")
    else:
        print(f"⚠️ 测试集为空，跳过预测")

# 3. 对单张图片进行预测示例
print("\n=== 3. 对单张图片进行预测示例 ===")

test_images = list((work_dir / 'Datas' / 'test' / 'images').glob('*'))
if test_images:
    sample_image = test_images[0]
    print(f"预测图片: {sample_image.name}")

    results = model.predict(
        source=str(sample_image),
        device='0',
        imgsz=640,
        conf=0.25,
        save=True,
        show=False,
        name='single_image_prediction',
        project=str(results_dir)
    )

    # 打印检测结果
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"检测到 {len(boxes)} 个对象")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                print(f"  对象 {i}: 类别={cls}, 置信度={conf:.3f}, 坐标={coords}")
        else:
            print("未检测到对象")

    print(f"✅ 预测结果已保存到: {results_dir / 'single_image_prediction'}")
else:
    print("⚠️ 测试集为空，跳过单张图片测试")

# 4. 总结
print(f"\n=== 测试完成！ ===")
print(f"所有测试结果保存在: {results_dir}")

# 显示目录结构
print(f"\n测试结果目录结构：")
for root, dirs, files in os.walk(results_dir):
    level = root.replace(str(results_dir), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... 等{len(files)}个文件")