import cv2
from ultralytics import YOLO
import numpy as np
from utils import *

PIXEL_THRESHOLD = 2000  # 設定閾值，僅保留像素數大於該值的區域
AREA_THRESHOLD = 500 # 設定閾值，避免過小的分割區域
DISTANCE_THRESHOLD = 250 # 定義距離閾值（例如：設定 10 為最大可接受距離）
SHORT_SIDE = 120 # 轉動短邊判斷閾值
TWO_POINT_TEETH_THRESHOLD = 259 # 初判單雙牙尖使用
RANGE_FOR_TOOTH_TIP_LEFT = 80 # 強迫判斷雙牙尖，中心區域定義使用(左)
RANGE_FOR_TOOTH_TIP_RIGHT = 40 # 強迫判斷雙牙尖，中心區域定義使用(右)

def less_than_area_threshold(component_mask, area_threshold):
    """根據指定面積大小，過濾過小的分割區域"""
    area = cv2.countNonZero(component_mask)
    # 去除掉過小的分割區域
    if area < area_threshold:
        return True
    return False

def get_mid_point(image, dilated_mask, idx):
    """取得物件中點，並且繪製點及idx標籤於 image"""
    non_zero_points = np.column_stack(np.where(dilated_mask > 0))
    if len(non_zero_points) > 0:
        mid_point = np.mean(non_zero_points, axis=0).astype(int)
        mid_y, mid_x = mid_point
        # 在繪製用圖片標註 dentin 中心點及對應數字標籤
        cv2.putText(image, str(idx), (mid_x-5, mid_y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, (mid_x, mid_y), 5, (255, 255, 0), -1)  # 黃色圓點
    return mid_y, mid_x

def locate_points_with_dental_crown(dental_crown_bin, dilated_mask, mid_x, mid_y, overlay):
    """處理與 dental_crown 之交點 (Enamel的底端)"""
    # 獲取每個獨立 mask 與原始 mask 的交集區域
    #breakpoint()
    intersection = cv2.bitwise_and(dental_crown_bin, dilated_mask)

    overlay[intersection > 0] = (255, 0, 0)  # 將 dentin 顯示
    # 取得交集區域的 contour 作為交點
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # 將交點進行排序
    corners = get_top_points(contours, reverse=True)
    # 確認排序成功成功
    if corners is not None:
        # 整數化
        corners = np.int32(corners)
        # 初始化取得座標，預設左右兩邊皆有
        enamel_left_x = None
        enamel_left_y = None
        enamel_right_x = None
        enamel_right_y = None
        # 針對每個點進行處理
        for corner in corners:
            # 取得交點座標
            x, y = corner.ravel()
            # 判斷左右
            if x < mid_x:
                # 後續判斷
                if enamel_left_x is not None:
                    # 找到 y 最大者
                    if y > enamel_left_y:
                        enamel_left_x = x
                        enamel_left_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, enamel_left_x):
                        continue
                # 初判
                else:
                    enamel_left_x = x
                    enamel_left_y = y
            elif x > mid_x:
                # 後續判斷
                if enamel_right_x is not None:
                    # 找到 y 最大者
                    if y > enamel_right_y:
                        enamel_right_x = x
                        enamel_right_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, enamel_right_x):
                        continue
                # 初判
                else:
                    enamel_right_x = x
                    enamel_right_y = y
    return enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y

def locate_points_with_gum(gum_bin, dilated_mask, mid_x, mid_y, overlay):
    """處理與 gum 之交點 (Alveolar_bone的頂端)"""
    # 獲取每個獨立 mask 與原始 mask 的交集區域
    intersection = cv2.bitwise_and(gum_bin, dilated_mask)
    overlay[intersection > 0] = (0, 255, 0)  # 將 dentin 顯示
    # 取得交集區域的 contour 作為交點
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # 反向排序
    corners = get_top_points(contours, reverse=False) #  
    # 確認排序成功
    if corners is not None:
        # 整數化
        top_corners = np.int32(corners)  
        # 初始化取得座標，預設左右兩邊皆有
        gum_left_x = None
        gum_left_y = None
        gum_right_x = None
        gum_right_y = None
        # 針對每個點進行處理
        for corner in top_corners:
            # 取得交點座標
            x, y = corner.ravel()
            # 如果該交點超出中心點太多，就過濾掉
            if x >= mid_x-40 and x <= mid_x+40:
                continue
            # 判斷左右
            if x < mid_x:
                # 後續判斷
                if gum_left_x is not None:
                    # 找到 y 最小者
                    if y < gum_left_y:
                        gum_left_x = x
                        gum_left_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, gum_left_x):
                        continue
                # 初判
                else:
                    gum_left_x = x
                    gum_left_y = y
            elif x > mid_x:
                # 後續判斷
                if gum_right_x is not None:
                    # 找到 y 最小者
                    if y < gum_right_y:
                        gum_right_x = x
                        gum_right_y = y
                        continue
                    # 因以排序，看到 x 座標過接近的交點就不要重複看
                    elif is_within_range(x, gum_right_x):
                        continue
                # 初判
                else:
                    gum_right_x = x
                    gum_right_y = y
    return gum_left_x, gum_left_y, gum_right_x, gum_right_y

def locate_points_with_dentin(gum_bin, dilated_mask, mid_x, mid_y, angle ,short_side, image, component_mask):
    # 取得 dentin 與 gum 交集
    intersection = cv2.bitwise_and(gum_bin, dilated_mask)
    # 由於希望取得 dentin 與 gum 交集的底部，所以需要轉正
    # 建立旋轉後的 mask 和 繪製用圖片
    c_image = rotate_image(image, angle)
    c_intersection = rotate_image(intersection, angle)
    c_dilated_mask = rotate_image(dilated_mask, angle)
    # 取得旋轉後區域的中點
    non_zero_points = np.column_stack(np.where(c_dilated_mask > 0))
    if len(non_zero_points) > 0:
        mid_point = np.mean(non_zero_points, axis=0).astype(int)
        c_mid_y, c_mid_x = mid_point
    # 初始化取得座標，預設左右兩邊皆有        
    dentin_left_x = None
    dentin_left_y = None
    dentin_right_x = None
    dentin_right_y = None
    # 根據短邊大小(寬度)，初步判斷單牙尖或雙牙尖
    if short_side > TWO_POINT_TEETH_THRESHOLD:
        # 較寬者，強迫判斷為雙牙尖
        contours, _ = cv2.findContours(c_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, None, None
        top_points = get_top_points(contours, reverse=True)
        bottom_corners = top_points
        # 初始化左右兩邊的點列表
        left_corners = []
        right_corners = []
        # 根據 c_mid_x 分配點到左右兩邊
        for point in bottom_corners:
            x, y = point.ravel() # 取得左右兩邊
            if x < c_mid_x-RANGE_FOR_TOOTH_TIP_LEFT:
                left_corners.append(point)
            # 雙牙尖，太中間的點不可能是牙尖
            elif x >= c_mid_x-RANGE_FOR_TOOTH_TIP_LEFT and x <= c_mid_x+RANGE_FOR_TOOTH_TIP_RIGHT:
                continue
            else:
                right_corners.append(point)
        # 左牙尖判斷
        for corner in left_corners:
            # 取得點座標
            x, y = corner.ravel()
            # 確定為左邊
            if x < c_mid_x:
                # 後續判斷
                if dentin_left_x is not None:
                    # 取得 y 最大者
                    if y > dentin_left_y:
                        dentin_left_x = x
                        dentin_left_y = y
                        continue
                # 初判
                else:
                    dentin_left_x = x
                    dentin_left_y = y
        # 右牙尖判斷
        for corner in right_corners:
            # 取得點座標
            x, y = corner.ravel()
            # 確定為右邊
            if x > c_mid_x:
                # 後續判斷
                if dentin_right_x is not None:
                    # 取得 y 最大者
                    if y > dentin_right_y:
                        dentin_right_x = x
                        dentin_right_y = y
                        continue
                # 初判
                else:
                    dentin_right_x = x
                    dentin_right_y = y
        # 避免 None 存在，因有可能在上述流程誤判為雙牙尖
        dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y)
        #---- 例外狀況 ---- 左右牙尖高度落差較大
        print("Debugging ::")
        if all(v is None for v in [dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y]):
            print("All variables are None.")
        else:
            if not is_within_range(dentin_left_y, dentin_right_y, 200):
                if dentin_right_y > dentin_left_y:
                    dentin_left_y = dentin_right_y
                    dentin_left_x = dentin_right_x
                else:
                    dentin_right_y = dentin_left_y
                    dentin_right_x = dentin_left_x
    else:
        # 進行交點搜尋
        contours, _ = cv2.findContours(c_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, None, None
        # 排序
        bottom_corners = get_top_points(contours, reverse=True)
        for corner in bottom_corners:
            x, y = corner.ravel()
            # 判斷左右邊(預設是雙牙尖)
            if x < c_mid_x:
                # 後續判斷
                if dentin_left_x is not None:
                    # 取得 y 最大者
                    if y > dentin_left_y:
                        dentin_left_x = x
                        dentin_left_y = y
                        continue
                # 初判
                else:
                    dentin_left_x = x
                    dentin_left_y = y
            elif x > c_mid_x:
                # 後續判斷
                if dentin_right_x is not None:
                    # 取得 y 最大者
                    if y > dentin_right_y:
                        dentin_right_x = x
                        dentin_right_y = y
                        continue
                # 初判
                else:
                    dentin_right_x = x
                    dentin_right_y = y
         # 避免 None 存在，因有可能在上述流程誤判為雙牙尖
        dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y)
        # 如果判斷出來的雙邊牙尖過於接近，確定應為單牙尖狀況，故指定最小者為牙尖
        if is_within_range(dentin_left_x, dentin_right_x, 80):
            bottom_corner = bottom_corners[:1]
            for corner in bottom_corner:
                x, y = corner.ravel()
                dentin_left_x = x
                dentin_left_y = y
                dentin_right_x = x
                dentin_right_y = y
                cv2.circle(c_image, (x, y), 5, (255, 0, 0), -1)
    
    # 將旋轉後的座標，轉回原角度
    dentin_left_coord = [dentin_left_x, dentin_left_y]
    dentin_right_coord = [dentin_right_x, dentin_right_y]
    dentin_left_x, dentin_left_y = convert_coord_back(dentin_left_coord, angle, component_mask)
    dentin_right_x, dentin_right_y = convert_coord_back(dentin_right_coord, angle, component_mask)
    
    # 膨脹避免資訊被截斷(旋轉回去有可能截到)
    kernel = np.ones((3, 3), np.uint8)
    c_dilated_mask = cv2.dilate(c_dilated_mask, kernel, iterations=5)
    c_dilated_mask_rotated_back = rotate_image(c_dilated_mask, -angle)
    c_image_rotated_back = rotate_image(c_image, -angle)
    mask = c_dilated_mask_rotated_back > 0
    # 把對應區域放回去原圖
    image[mask] = c_image_rotated_back[mask]
    
    return dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y
def locate_points(image, component_mask, binary_images, idx, overlay):
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

def draw_image_and_print_information(prediction, image_for_drawing, line_image):
    # 繪圖及印出資訊
    print("Mid Points : ", prediction["mid"])
    print("enamel_left : ", prediction["enamel_left"])
    print("enamel_right : ", prediction["enamel_right"])
    print("gum_left : ", prediction["gum_left"])
    print("gum_right : ", prediction["gum_right"])
    print("dentin_left : ", prediction["dentin_left"])
    print("dentin_right : ", prediction["dentin_right"])
    cv2.circle(image_for_drawing, prediction["enamel_left"], 5, (0, 0, 255), -1)
    cv2.circle(image_for_drawing, prediction["enamel_right"], 5, (0, 0, 255), -1)
    cv2.circle(image_for_drawing, prediction["gum_left"], 5, (0, 255, 0), -1)
    cv2.circle(image_for_drawing, prediction["gum_right"], 5, (0, 255, 0), -1)
    cv2.circle(image_for_drawing, prediction["dentin_left"], 5, (255, 0, 0), -1)
    cv2.circle(image_for_drawing, prediction["dentin_right"], 5, (255, 0, 0), -1)
    # Draw lines between points
    print("e_l -> d_l : ", prediction["enamel_left"], prediction["dentin_left"])
    if (prediction["enamel_left"][0] is not None) and (prediction["dentin_left"] is not None):
        cv2.line(line_image, prediction["enamel_left"], prediction["dentin_left"], (0, 0, 255), 2)
    else:
        print("None Detected. Not drawing line.")
        
    print("e_l -> g_l : ", prediction["enamel_left"], prediction["gum_left"])
    if (prediction["enamel_left"][0] is not None) and (prediction["gum_left"][0] is not None):
        cv2.line(line_image, prediction["enamel_left"], prediction["gum_left"], (0, 255, 255), 2)
    else:
        print("None Detected. Not drawing line.")
        
    print("e_r -> d_r : ", prediction["enamel_right"], prediction["dentin_right"])
    if (prediction["enamel_right"][0] is not None) and (prediction["dentin_right"] is not None):
        cv2.line(line_image, prediction["enamel_right"], prediction["dentin_right"], (0, 0, 255), 2)
    else:
        print("None Detected. Not drawing line.")
    
    print("e_r -> g_r : ", prediction["enamel_right"], prediction["gum_right"])
    if (prediction["enamel_right"][0] is not None) and (prediction["gum_right"][0] is not None):
        cv2.line(line_image, prediction["enamel_right"], prediction["gum_right"], (0, 255, 255), 2)
    else:
        print("None Detected. Not drawing line.")

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
        # Get class ID and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)
        
        # Get class name
        class_name = class_names[class_id]

        # Convert mask to numpy array and resize to match original image
        mask_np = mask.cpu().numpy()
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        
        # Convert mask to binary image
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        if class_name not in masks_dict.keys():
            masks_dict[class_name]=mask_binary
        else:
            masks_dict[class_name]=np.maximum(masks_dict[class_name], mask_binary) #find the union of masks
            #print('repeat!!!')
        # Check if the mask is valid
        if np.sum(mask_binary) == 0:
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
            masks_dict['dentin']=np.maximum(masks_dict['dentin'], masks_dict[key]) # find the union
            
    dental_contours=np.maximum(masks_dict['dentin'], masks_dict['dental_crown'])
 
    kernel = np.ones((30,30), np.uint8)
    filled = cv2.morphologyEx(dental_contours, cv2.MORPH_CLOSE, kernel)
    # contours, _ = cv2.findContours(dental_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 創建空白圖像
    # filled = np.zeros_like(dental_contours)
    # # 填充所有輪廓
    # cv2.drawContours(filled, contours, -1, (255,255,255), -1)
    filled=cv2.bitwise_and(filled, cv2.bitwise_not(masks_dict['dental_crown']))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    masks_dict['dentin']=filled
    
    
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
        draw_image_and_print_information(prediction, image_for_drawing, line_image)    

    breakpoint()
