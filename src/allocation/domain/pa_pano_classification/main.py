
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np 
import torch.nn.functional as F  # 添加這行
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 根據圖像大小調整
        self.fc2 = nn.Linear(128, 2)  # 二分類

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def create_pa_pano_classification_model(model_path):
    model = SimpleCNN()
    if torch.cuda.is_available():  # 如果有可用的GPU
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
def read_pil_image(image_path):
    # 使用 cv2 加載圖片
    return Image.open(image_path).convert('RGB')

def predict_image_pa_pano_classification(model, image):
    predicted_class_mapping={
        0: "periapical film",
        1: "panoramic x-ray",
    }
    # 設定設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定義與訓練時相同的轉換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 調整圖片大小
        transforms.ToTensor(),           # 轉為Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])

    image = transform(image).unsqueeze(0)  # 增加一個維度以符合模型輸入格式 (batch_size, channels, height, width)
    image = image.to(device)  # 將圖片移到設備上
    
    # 將模型設置為評估模式
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)  # 獲取模型輸出
        _, predicted = torch.max(outputs.data, 1)  # 獲取預測類別
        scores = torch.sigmoid(outputs)
    
    return predicted_class_mapping[predicted.item()], scores.squeeze().cpu().numpy()[predicted.item()]

if __name__ == '__main__':



    model = create_pa_pano_classification_model("./models/pa_pano_classification.pth")  # 載入最佳模型的權重

    # 測試單張圖片
    image_path = './tests/files/027107.jpg'  # 替換為你的圖片路徑
    image = read_pil_image(image_path)
    predicted_class, scores = predict_image_pa_pano_classification(model, image)
    
    print(f'Predicted class: {predicted_class}')
    print(f'Scores: {scores}')
    