import cv2
from ultralytics import YOLO
import numpy as np
from src.allocation.domain.dental_measure.utils import *
components_model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
contour_model=YOLO('./models/dentistryContour_yolov11n_4.46.pt')

def extract_features(masks_dict, original_img):
    """從遮罩中提取特徵點與區域資訊"""
    overlay = original_img.copy()
    line_image = original_img.copy()
    kernel = np.ones((3, 3), np.uint8)

    # 清理各個遮罩
    masks_dict['dental_crown'] = clean_mask(masks_dict['dental_crown'])
    masks_dict['dentin'] = clean_mask(masks_dict['dentin'], kernel_size=(30, 1), iterations=1)
    # binary_images['gum'] = clean_mask(binary_images['gum'], kernel_size=(30, 1), iterations=2)

    # 保留最大區域
    masks_dict['gum'] = extract_largest_component(masks_dict['gum'])

    # 膨脹處理後的 gum
    masks_dict['gum'] = cv2.dilate(masks_dict['gum'], kernel, iterations=10)

    # 合併所有遮罩
    #combined_mask = combine_masks(dilated_gum, binary_images)
    combined_mask = combine_masks(masks_dict)
    non_masked_area = cv2.bitwise_not(combined_mask)

     # 繪製 overlay
    # overlay[binary_images["dental_crown"] > 0] = (163, 118, 158)  # 將 dental_crown 顯示
    # overlay[binary_images["dentin"] > 0] = (117, 122, 152)  # 將 dentin 顯示
    # overlay[binary_images['gum'] > 0] = (0, 177, 177)  # 將 dentin 顯示
    # overlay[binary_images['crown'] > 0] = (255, 0, 128) # 將 crown 顯示
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
    for key, (x, y) in prediction.items():
        # 對每一個座標進行 safe_int 處理
        prediction[key] = (int_processing(x), int_processing(y))
    return prediction

def get_mask_dict_from_model(model, image, method='semantic'):
    results=model.predict(image)
    result=results[0]

    detections=[]
    # 獲取類別名稱
    class_names = result.names
    class_name_list=[]         
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs

    if masks is None:
        return {}
    
    masks_dict={}
    for mask, box in zip(masks.data, boxes):
        class_id = int(box.cls)# Get class ID and confidence
        #confidence = float(box.conf)=
        class_name = class_names[class_id] # Get class name
        mask_np = mask.cpu().numpy() # Convert mask to numpy array and resize to match original image
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255 # Convert mask to binary image

        if np.sum(mask_binary) == 0: #error handling
            continue

        if method=='semantic':
            if class_name not in masks_dict.keys():
                masks_dict[class_name]=mask_binary
            else:
                masks_dict[class_name]=cv2.bitwise_or(masks_dict[class_name], mask_binary) #find the union of masks
        else:
            if class_name not in masks_dict.keys():
                masks_dict[class_name]=[mask_binary]
            else:
                masks_dict[class_name].append(mask_binary)           
            

    return masks_dict

def generate_error_image(text):
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # 設置文字內容
    #text = "請上傳正確的圖片"

    # 設置字體、大小和顏色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # 白色
    thickness = 2

    # 計算文字大小，返回的是寬度和高度
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 計算文字放置的起始位置，使其位於圖像的中心
    text_x = (image.shape[1] - text_width) // 2
    text_y = (image.shape[0] + text_height) // 2

    # 在圖像上添加文字
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    return image

def dental_estimation(image, scale=(31/960,41/1080), return_type='image'):
    print(return_type)
    components_model_masks_dict=get_mask_dict_from_model(components_model, image, method='semantic')
    contours_model_masks_dict=get_mask_dict_from_model(contour_model, image, method='instance')

    denti_measure_names_map={
        'Alveolar_bone': 'gum',
        'Dentin': 'dentin',
        'Enamel': 'dental_crown',
        'Crown': 'crown' ### In fact it is enamel (why labeling man so stupid)
    }
    components_model_masks_dict = {denti_measure_names_map.get(k, k): v for k, v in components_model_masks_dict.items()}

    # Error handling

    required_components = {
        'dentin': "No dental instance detected",
        'dental_crown': "No dental_crown detected",
        'gum': "No gum detected"
    }
    for component, error_message in required_components.items():
        if components_model_masks_dict.get(component) is None:
            if return_type=='image':
                return generate_error_image(error_message)
            else:
                return []
        
    if contours_model_masks_dict.get('dental_contour') is None:
        generate_error_image("No dental instance detected")
        if return_type=='image':
            return generate_error_image(error_message)
        else:
            return []
        
    # dentin is compensated by dental contour model
    for mask in contours_model_masks_dict['dental_contour']:
        if not 'dental_contour' in components_model_masks_dict.keys():
            components_model_masks_dict['dental_contour']=mask
        else:
            components_model_masks_dict['dental_contour']=cv2.bitwise_or(components_model_masks_dict['dental_contour'], mask) # find the union
    crown_or_enamal_mask=np.zeros_like(components_model_masks_dict['dentin'])
    for key in ['dental_crown','crown']:
        if components_model_masks_dict.get(key) is not None:
            crown_or_enamal_mask=cv2.bitwise_or(crown_or_enamal_mask, components_model_masks_dict[key])
    components_model_masks_dict['dentin']=components_model_masks_dict['dental_contour']-cv2.bitwise_and(components_model_masks_dict['dental_contour'], crown_or_enamal_mask)


    overlay, line_image, non_masked_area= extract_features(components_model_masks_dict, image) # 處理繪圖用圖片等特徵處理後圖片

    predictions = []
    image_for_drawing=image.copy()
    #for i in range(1, num_labels):  # 從1開始，0是背景

    for i, component_mask in enumerate(contours_model_masks_dict['dental_contour']):
        # 取得分析後的點
        prediction = locate_points(image_for_drawing, component_mask, components_model_masks_dict, i+1, overlay)
        # 如果無法判斷點，會回傳空字典
        if len(prediction) == 0:
            continue
        prediction["teeth_id"] = i
        prediction["dentin_id"] = None
        print(f"Tooth {i}")
        image_for_drawing=draw_point(prediction, image_for_drawing)
        image_for_drawing, dental_pair_list=draw_line(prediction, image_for_drawing, scale)
        prediction['pair_measurements']=dental_pair_list
        if dental_pair_list:
            predictions.append(prediction)

    if return_type=='image':
        return image_for_drawing
    else:

        return predictions

