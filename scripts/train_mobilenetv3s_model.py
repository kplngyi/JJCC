import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA_ROOT = "/Users/hpyi/Hobby/JJCC/data"
BATCH_SIZE = 8
EPOCHS = 30
IMG_SIZE = 90  # 将棋子缩放到 90x90 足够识别

# 2. 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2), # 模拟光影变化
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 打印类别对应关系 (非常重要，后面推理要用到)
class_names = train_dataset.classes
print(f"检测到类别: {class_names}")

# 3. 定义模型 (使用预训练的 MobileNetV3-Small)
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
# 修改最后的全连接层以匹配你的 15 个类别 (14种棋子 + None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model = model.to(DEVICE)

# 4. 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. 训练循环
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # 每个 Epoch 验证一次
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {100 * correct / total:.2f}%")

# 6. 保存模型
save_path = "/Users/hpyi/Hobby/JJCC/assets/chess_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, save_path)
print("模型已保存至 assets/chess_model.pth")

def visualize_misclassified(model, val_loader, class_names, device, max_per_class=5):
    """
    可视化验证集上预测错误的图片
    """
    model.eval()
    misclassified = {cls: [] for cls in class_names}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    img_path = val_loader.dataset.samples[i][0]
                    true_label = class_names[labels[i]]
                    pred_label = class_names[predicted[i]]
                    misclassified[true_label].append((img_path, pred_label))

    # 可视化
    for cls, imgs in misclassified.items():
        if len(imgs) == 0:
            continue
        print(f"\n类别 '{cls}' 的预测错误示例（最多 {max_per_class} 张）:")
        for img_path, pred_label in imgs[:max_per_class]:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"True: {cls}, Pred: {pred_label}")
            plt.show()
visualize_misclassified(model, val_loader, class_names, DEVICE, max_per_class=3)