import cv2
from ultralytics import YOLO
import numpy as np
from utils import *


def extract_features(masks_dict, original_img):
    """從遮罩中提取特徵點與區域資訊"""
    overlay = original_img.copy()
    line_image = original_img.copy()
    kernel = np.ones((3, 3), np.uint8)

    # 清理各個遮罩
    masks_dict['dental_crown'] = clean_mask(masks_dict['dental_crown'])
    masks_dict['dentin'] = clean_mask(masks_dict['dentin'], kernel_size=(30, 1), iterations=1)
    masks_dict['gum'] = clean_mask(masks_dict['gum'], kernel_size=(30, 1), iterations=2)

    # 保留最大區域
    masks_dict['gum'] = extract_largest_component(masks_dict['gum'])

    # 膨脹處理後的 gum
    masks_dict['gum'] = cv2.dilate(masks_dict['gum'], kernel, iterations=10)
    
    # 膨脹處理dentin
    dental_contours=np.maximum(masks_dict['dentin'], masks_dict['dental_crown'])    
    kernel = np.ones((23,23), np.uint8)
    filled = cv2.morphologyEx(dental_contours, cv2.MORPH_CLOSE, kernel)
    filled=cv2.bitwise_and(filled, cv2.bitwise_not(masks_dict['dental_crown']))
    masks_dict['dentin']=filled
    
    # 合併所有遮罩
    combined_mask = combine_masks(masks_dict)
    non_masked_area = cv2.bitwise_not(combined_mask)

     # 繪製 overlay
    key_color_mapping={
        'dental_crown': (163, 118, 158),
        'dentin':(117, 122, 152),
        'gum': (0, 177, 177),
        'crown': (255, 0, 128),
    }
    for key in key_color_mapping.keys():
        if masks_dict.get(key) is not None:
            overlay[masks_dict[key] > 0] = key_color_mapping[key]

    # 回傳疊加後的影像和線條影像
    return overlay, line_image, non_masked_area


def locate_points(image, component_mask, binary_images, idx, overlay):
    def less_than_area_threshold(component_mask, area_threshold):
        """根據指定面積大小，過濾過小的分割區域"""
        area = cv2.countNonZero(component_mask)
        # 去除掉過小的分割區域
        if area < area_threshold:
            return True
        return False    
    
    """以分割後的 dentin 為單位進行處理"""
    prediction = {}
    if less_than_area_threshold(component_mask, AREA_THRESHOLD):
        return prediction
    # 以方框框住該 component_mask，整數化
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0]) # 最小區域長方形
    box = cv2.boxPoints(rect) # 取得長方形座標
    box = np.int32(box) # 整數化 
    width = rect[1][0]  # 寬度
    height = rect[1][1]  # 高度
    short_side = min(width, height)  # 短邊
    long_side = max(width, height)
    if short_side < SHORT_SIDE:
       return prediction
    
    # 判斷旋轉角度
    angle = get_rotation_angle(component_mask)
    # 如果長短邊差距在 30 內 (接近正方形)，不轉動
    if is_within_range(short_side, long_side, 30):
        angle = 0
        
    # 膨脹獨立 dentin 分割區域
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(component_mask, kernel, iterations=7)
 
    # 取得中點
    mid_y, mid_x = get_mid_point(image, dilated_mask, idx)

    ########### 處理與 dental_crown 之交點 (Enamel的底端) ########### 
    enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y = locate_points_with_dental_crown(binary_images["dental_crown"], dilated_mask, mid_x, mid_y, overlay)
    ########### 處理與 gum 之交點 (Alveolar_bone的頂端) ########### 
    gum_left_x, gum_left_y, gum_right_x, gum_right_y = locate_points_with_gum(binary_images["gum"], dilated_mask, mid_x, mid_y, overlay)
    ########### 處理與 dentin 的底端 ########### 
    dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = locate_points_with_dentin(binary_images["gum"], dilated_mask, mid_x, mid_y, angle, short_side, image, component_mask)
    
    prediction = {"mid": (mid_x, mid_y), 
                "enamel_left": (enamel_left_x, enamel_left_y), "enamel_right":(enamel_right_x, enamel_right_y),
                "gum_left":(gum_left_x, gum_left_y), "gum_right": (gum_right_x, gum_right_y),
                "dentin_left":(dentin_left_x, dentin_left_y), "dentin_right":(dentin_right_x, dentin_right_y),
                }
    return prediction



if __name__ == '__main__':
    image=cv2.imread('./data/pics/raw_12.png')
    model=YOLO('./model/dentistry_yolov11x-seg_4.42.pt')
    results=model.predict(image)
    result=results[0]

    detections=[]
    # 獲取類別名稱
    class_names = result.names
    class_name_list=[]         
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs

    masks_dict={}
    for mask, box in zip(masks.data, boxes):
        class_id = int(box.cls)# Get class ID and confidence
        confidence = float(box.conf)
        
        class_name = class_names[class_id] # Get class name

        mask_np = mask.cpu().numpy() # Convert mask to numpy array and resize to match original image
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255 # Convert mask to binary image

        if class_name not in masks_dict.keys():
            masks_dict[class_name]=mask_binary
        else:
            masks_dict[class_name]=cv2.bitwise_or(masks_dict[class_name], mask_binary) #find the union of masks
            
        if np.sum(mask_binary) == 0: #error handling
            continue

    #_, binary_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    denti_measure_names_map={
        'Alveolar_bone': 'gum',
        'Dentin': 'dentin',
        'Enamel': 'dental_crown',
        'Crown': 'crown'
    }

    masks_dict = {denti_measure_names_map.get(k, k): v for k, v in masks_dict.items()}

    #dentin=dentin+pulp+restoration
    for key in ['Pulp','Restoration']:
        if masks_dict.get(key) is not None:
            masks_dict['dentin']=cv2.bitwise_or(masks_dict['dentin'], masks_dict[key]) # find the union

    #dental crown=caries+crown   
    for key in ['Caries']:
        if masks_dict.get(key) is not None:
            masks_dict['dental_crown']=cv2.bitwise_or(masks_dict['dental_crown'], masks_dict[key])


    overlay, line_image, non_masked_area= extract_features(masks_dict, image) # 處理繪圖用圖片等特徵處理後圖片

    num_labels, labels = cv2.connectedComponents(masks_dict['dentin']) # 取得分割開的 dentin , num_labels 表示 labels 數量， labels 則是分割對應

    # 針對獨立分割 dentin 個別處理
    predictions = []
    image_for_drawing=image.copy()
    for i in range(1, num_labels):  # 從1開始，0是背景
        component_mask = np.uint8(labels == i) * 255
        # 取得分析後的點
        prediction = locate_points(image_for_drawing, component_mask, masks_dict, i, overlay)
        # 如果無法判斷點，會回傳空字典
        if len(prediction) == 0:
            continue
        prediction["tooth_id"] = i
        prediction["dentin_id"] = None
        predictions.append(prediction)
        print(f"Tooth {i}")
        image_for_drawing=draw_point(prediction, image_for_drawing)
        image_for_drawing=draw_line(prediction, image_for_drawing)
            
    show_two(overlay,image_for_drawing)
    breakpoint()
    #for prediction in predictions:
        
    #polt for labels
