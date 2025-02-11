import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import glob
import cv2
from sklearn.cluster import DBSCAN
def show_plot(image):
    """
    顯示圖片並顯示滑鼠當前指向的 XY 座標

    :param image: 要顯示的圖片
    """
    # 顯示圖片
    fig, ax = plt.subplots(figsize=(8, 6))  # 設定畫布大小
    ax.imshow(image)  # 顯示圖片
    ax.axis('off')  # 不顯示坐標軸
    ax.set_title("Image Display")  # 圖片標題

    # 定義顯示座標的函數
    def on_mouse_move(event):
        if event.xdata and event.ydata:  # 檢查是否有效的座標
            ax.set_title(f"X: {event.xdata:.2f}, Y: {event.ydata:.2f}")
            fig.canvas.draw()

    # 連接滑鼠移動事件
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # 顯示圖片
    plt.show()
def create_image_text_dict(val_img_path, txt_files_path):
    # 列出所有的圖片文件
    image_files = [f for f in os.listdir(val_img_path) if f.endswith(('.png', '.jpg', '.tmp'))]
    
    # 初始化字典
    image_text_dict = {}

    for image_file in image_files:
        # 去掉文件擴展名以獲取基礎名稱
        base_name = os.path.splitext(image_file)[0]
        # 構建對應的txt文件名
        txt_file_name = f"{base_name}.txt"
        txt_file_path = os.path.join(txt_files_path, txt_file_name)
        
        # 檢查txt文件是否存在
        if os.path.isfile(txt_file_path):
            # with open(txt_file_path, 'r') as txt_file:
            #     # 讀取txt文件內容
            #     content = txt_file.read()
            #     # 將圖片文件名和對應的txt內容添加到字典中
            image_text_dict[base_name] = {'image_path': os.path.join(val_img_path, image_file), 'txt_path': txt_file_path}

    return image_text_dict

def generate_gt_mask_from_txt(txt_path, img_shape):
    # 初始化一個空的 GT mask
    height, width = img_shape
    gt_mask_dict={}
    with open(txt_path, 'r') as file:
        for line in file.readlines():
            # 解析每一行，獲取頂點坐標
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            coords = list(map(float, line.strip().split()))
            class_id = int(coords[0])  # 類別ID（可選）
            points = []

            # 將相對坐標轉換為絕對坐標
            for i in range(1, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i + 1] * height)
                points.append((x, y))

            # 將點轉換為 NumPy 數組
            points = np.array(points, dtype=np.int32)
            db = DBSCAN(eps=100, min_samples=30)
            labels = db.fit_predict(points)
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                cluster_points = points[labels == cluster_id]
                points_poly = cluster_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(gt_mask, [points_poly], 255)                                                          

            gt_mask_dict[class_id]=gt_mask
    return gt_mask_dict

if __name__=='__main__':
    model_path='./models/dentistry_yolov11x-seg-all_4.42.pt'
    model=YOLO(model_path)

    val_img_path='./datasets/split_data_4.42/images/val'
    val_label_path='./datasets/split_data_4.42/labels/val'

    image_text_dict=create_image_text_dict(val_img_path, val_label_path)
    for _, image_info_dict in image_text_dict.items():
        image_path=image_info_dict['image_path']
        txt_path=image_info_dict['txt_path']
        image=cv2.imread(image_path)
        mask_dict=generate_gt_mask_from_txt(txt_path, image.shape[:2])
        breakpoint()

    