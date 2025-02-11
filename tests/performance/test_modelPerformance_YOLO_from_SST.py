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

def get_files_dict(directory):
    file_paths = []
    # 使用 os.walk 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取完整的文件路径
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    #build the map
    file_map = {}
    image_extensions=['.jpg', '.jpeg', '.png', '.bmp']
    for file in file_paths:
        filename, extension = os.path.splitext(file)
        base_name = os.path.splitext(os.path.basename(file))[0]
        if base_name not in file_map:
            file_map[base_name] = {}
        if extension in image_extensions:
            file_map[base_name]['image'] = file
        elif extension == '.json':
            file_map[base_name]['json'] = file
        elif extension == '.npy':
            file_map[base_name]['npy'] = file  
    return file_map
def match_sst_dict(val_img_path, file_name_dict):
    # 列出所有的圖片文件
    image_files = [f for f in os.listdir(val_img_path) if f.endswith(('.png', '.jpg', '.tmp'))]
    
    # 初始化字典
    image_text_dict = {}

    for image_file in image_files:
        # 去掉文件擴展名以獲取基礎名稱
        
        base_name = os.path.splitext(image_file)[0]
        if base_name in file_name_dict.keys():
            image_text_dict[base_name]=file_name_dict[base_name]

    return image_text_dict
if __name__=='__main__':
    model_path='./models/dentistry_yolov11x-seg-all_4.42.pt'
    model=YOLO(model_path)

    val_img_path='./datasets/split_data_4.42/images/val'
    dentistry_sst_path='./datasets/dentistry_4.40'
    all_file_name_dict=get_files_dict(dentistry_sst_path)
    val_file_name_dict=match_sst_dict(val_img_path, all_file_name_dict)
    breakpoint()


    