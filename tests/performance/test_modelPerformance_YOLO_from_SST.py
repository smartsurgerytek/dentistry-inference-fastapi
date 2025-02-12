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
import json
from src.allocation.domain.dental_segmentation.main import yolo_transform
from src.allocation.domain.dental_measure.utils import show_two
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

def load_image_and_mask(image_path, json_path, npy_path):

    label_info = {}
    image = None
    
    def handle_load_error(e):
        print(f"Error loading data: {e}")
        return label_info, np.array([]), False


    image = cv2.imread(image_path)

    # If the image still isn't loaded, try another method for paths with Chinese characters
    if image is None:
        try:
            image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image could not be loaded.")
        except (IOError, ValueError) as e:
            return handle_load_error(e)

    # Initialize the mask with the correct dimensions
    raw_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Load the JSON file and the mask
    try:
        with open(json_path, 'r') as f:
            label_info = json.load(f)
        
        raw_mask = np.load(npy_path)
    except (IOError, ValueError, json.JSONDecodeError) as e:
        return handle_load_error(e)
    return label_info, raw_mask, True

def retrive_gt_masks(label_info, raw_mask):
    #semantic mask
    semantic_mask_dict={}
    for index, key in label_info.items():
        if semantic_mask_dict.get(key) is not None:
            semantic_mask_dict[key]=cv2.bitwise_or(semantic_mask_dict[key], (raw_mask==int(index)).astype(np.uint8)*255)
        else:
            semantic_mask_dict[key]=(raw_mask==int(index)).astype(np.uint8)
    return semantic_mask_dict

def calculate_iou(mask1, mask2):
    """
    計算兩個二值掩碼之間的IoU（Intersection over Union）

    :param mask1: 第一個二值掩碼 (numpy array)
    :param mask2: 第二個二值掩碼 (numpy array)
    :return: IoU值
    """
    # 確保掩碼是二值數組
    if mask1.shape != mask2.shape:  # 確保兩個掩碼的尺寸相同
        raise ValueError("The masks must have the same shape.")
    
    # 將掩碼轉換為二值數組
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    # 計算交集和聯集
    intersection = np.sum(np.logical_and(mask1, mask2))  # 交集
    union = np.sum(np.logical_or(mask1, mask2))          # 聯集

    # 計算IoU
    iou = intersection / union if union != 0 else 0  # 防止除以零
    return iou

def plot_confusion_matrix(confusion_matrix, index_label_mapping):
    # 获取标签
    labels = [index_label_mapping[i] for i in range(len(confusion_matrix))]
    
    # 创建一个 Seaborn 热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, ha='right')  # 可选：旋转 x 轴标签，避免重叠
    plt.yticks(rotation=0)  # 可选：让 y 轴标签平行显示
    plt.tight_layout()
    plt.show()


def update_estimated_result(confusion_matrix, iou_label_dict, gt_masks_dict, pred_mask_dict, label_index_mapping):
    for gt_key, gt_mask in gt_masks_dict.items():
        if gt_key=='Background':
            continue
        gt_index=label_index_mapping[gt_key]
        if gt_key not in pred_mask_dict.keys():
            iou_label_dict[gt_key].append(0)
            confusion_matrix[gt_index, -1]+=1
            continue
        for pred_key, pred_mask in pred_mask_dict.items():
            if pred_key=='Background':
                continue
            pred_index=label_index_mapping[pred_key]
            iou_value=calculate_iou(gt_mask, pred_mask)
            if iou_value>0.5:
                confusion_matrix[gt_index, pred_index]+=1
            if gt_key==pred_key:
                iou_label_dict[gt_key].append(iou_value)
                if iou_value<0.5:
                    confusion_matrix[gt_index, -1]+=1

    for pred_key, _ in pred_mask_dict.items():
        if pred_key=='Background':
            continue
        if pred_key not in gt_masks_dict.keys():
            confusion_matrix[-1, pred_index]+=1
    return confusion_matrix, iou_label_dict
if __name__=='__main__':
    model_path='./models/dentistry_yolov11x-seg-all_4.42.pt'
    model=YOLO(model_path)

    val_img_path='./datasets/split_data_4.42/images/val'
    dentistry_sst_path='./datasets/dentistry_4.40'
    all_file_name_dict=get_files_dict(dentistry_sst_path)
    val_file_name_dict=match_sst_dict(val_img_path, all_file_name_dict)

    iou_label_dict = {
        "Alveolar_bone": [],
        "Caries": [],
        "Crown": [],
        "Dentin": [],
        "Enamel": [],
        "Implant": [],
        "Mandibular_alveolar_nerve": [],
        "Maxillary_sinus": [],
        "Periapical_lesion": [],
        "Post_and_core": [],
        "Pulp": [],
        "Restoration": [],
        "Root_canal_filling": [],
        "None": [],
    }
    label_index_mapping={}
    for idx, key in enumerate(iou_label_dict):
        label_index_mapping[key] = idx
    index_label_mapping={}
    for idx, key in enumerate(iou_label_dict):
        index_label_mapping[idx] = key

    len_label=len(iou_label_dict)
    confusion_matrix= np.zeros((len_label,len_label))

    for _, file_name_dict in val_file_name_dict.items():
        image_path=file_name_dict['image']
        json_path=file_name_dict['json']
        npy_path=file_name_dict['npy']
        # get the ground turth mask
        label_info, raw_mask, coexist_bool= load_image_and_mask(image_path, json_path, npy_path)
        image=cv2.imread(image_path)
        if not coexist_bool:
            breakpoint()
            continue        
        gt_masks_dict=retrive_gt_masks(label_info, raw_mask)
        # get the predicted mask
        pred_masks_dict=yolo_transform(image, model, return_type='dict', config=None, tolerance=0.5)

        confusion_matrix, iou_label_dict=update_estimated_result(confusion_matrix, 
                                                                 iou_label_dict, 
                                                                 gt_masks_dict, 
                                                                 pred_masks_dict, 
                                                                 label_index_mapping)
        #plot_confusion_matrix(confusion_matrix, index_label_mapping)

        
    #conpute the average of iou
    ave_iou_dict={}
    for key, iou_list in iou_label_dict.items():
        ave_iou_dict[key]=np.mean(iou_list)
    print(ave_iou_dict)
    plot_confusion_matrix(confusion_matrix, index_label_mapping)

    #
    breakpoint()

        


        


    