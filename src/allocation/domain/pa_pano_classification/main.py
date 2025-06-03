
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
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # 對應訓練時 image_size = 224
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def create_pa_pano_classification_model(model_path):
    model = SimpleCNN(num_classes=3)
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
        2: "other",
    }
    # 設定設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定義與訓練時相同的轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 訓練時 image_size = 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
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



    model = create_pa_pano_classification_model("./models/dentistry_pa-pano-classification_cnn_25.22.pth")  # 載入最佳模型的權重

    # 測試單張圖片
    image_path = './tests/files/027107.jpg'  # 替換為你的圖片路徑
    image = read_pil_image(image_path)
    predicted_class, scores = predict_image_pa_pano_classification(model, image)
    
    print(f'Predicted class: {predicted_class}')
    print(f'Scores: {scores}')
    