import sys
import os
from pathlib import Path
sys.path.insert(0, '...')
from ultralytics import YOLO


data_root = Path('...')


for split in ['train', 'val', 'test']:
    labels_dir = data_root / split / 'labels'
    images_dir = data_root / split / 'images'
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n{split}集：")
    print(f"图像数量：{len(image_files)}")
    print(f"标签数量：{len(os.listdir(labels_dir))}")
    
    
    for img_file in image_files:
        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        label_path = labels_dir / label_file
        
        if not label_path.exists() or os.path.getsize(label_path) <= 1:
            
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 1.0 1.0\n')
    
    print(f"修复后标签数量：{len(os.listdir(labels_dir))}")

print("\n=== 2. 加载YOLOv11Y模型 ===")
model = YOLO('best.pt', task='detect')


print("\n=== 3. 训练模型 ===")
results = model.train(
    data=str(data_root / 'dataset.yaml'),  
    epochs=2000,  
    batch=16,  
    imgsz=640,  
    device=0,  
    workers=16,  
    name='yolov11Y',  
    pretrained=True,  
    amp=True,  
    val=True  
)
print("\n=== 4. 评估模型 ===")
metrics = model.val()
print("\n=== 5. 保存模型 ===")
model.save('yolov11Y')
print('\n训练完成！')