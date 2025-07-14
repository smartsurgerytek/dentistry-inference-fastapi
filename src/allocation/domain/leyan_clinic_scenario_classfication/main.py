
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np 
import torch.nn.functional as F  # 添加這行
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# ========== 模型定義 ==========
class AngleAwareCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.angle_branch = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        self.classifier = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        angle_feat = self.angle_branch(x)
        return self.classifier(angle_feat)
    
def create_Leyan_clinic_scenario_classfication(model_path):
    model = AngleAwareCNN(num_classes=14)
    if torch.cuda.is_available():  # 如果有可用的GPU
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
def read_pil_image(image_path):
    # 使用 cv2 加載圖片
    return Image.open(image_path).convert('RGB')

# main.py 的修正後函式
def predict_image_Leyan_clinic_scenario_classfication(model, image):
    """
    預測圖片的臨床情境，並回傳最高分的類別以及所有類別的分數字典。
    """
    predicted_class_mapping = {
        0: "close", 1: "gag_bite", 2: "gag_open", 3: "left", 4: "left_45",
        5: "left_90", 6: "lower", 7: "open", 8: "other", 9: "right",
        10: "right_45", 11: "right_90", 12: "smile", 13: "top"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        
        # 1. 計算所有類別的機率 (使用 softmax 或 sigmoid)
        #    如果您的模型最後一層是 nn.Linear，通常用 softmax 進行多分類機率轉換
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        
        # 2. 找到分數最高的類別
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = predicted_class_mapping[predicted_idx]
        
        # 3. 建立包含所有類別和分數的字典
        all_scores = {predicted_class_mapping[i]: prob.item() for i, prob in enumerate(probabilities)}
        
        # 4. 回傳預測類別和包含所有分數的字典
        return predicted_class, all_scores


if __name__ == '__main__':



    model = create_Leyan_clinic_scenario_classfication("./models/dentistry_leyan_clinic-classification_cnn_25.28.pth")  # 載入最佳模型的權重

    # 測試單張圖片
    image_path = './tests/files/027107.jpg'  # 替換為你的圖片路徑
    image = read_pil_image(image_path)
    predicted_class, scores = predict_image_Leyan_clinic_scenario_classfication(model, image)
    
    print(f'Predicted class: {predicted_class}')
    print(f'Scores: {scores}')
    