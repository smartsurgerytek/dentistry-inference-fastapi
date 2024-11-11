import cv2
import numpy as np
import os
from utils import *

def find_the_extreme_points(input_points, count_limit):
    count=0
    max_y=0
    min_y=max(input_points, key=lambda x: x[1])[1]
    x=0
    #count_limit=400
    
    local_maximum=[]
    local_minimum=[]
    index=0
    find_max=True
    while x!=input_points[-1][0]:
        x, y= input_points[count+index]
        if find_max and y>=max_y:
            max_y=y
            max_y_point=(x,y)
            max_index=count+index
        elif not find_max and y<=min_y:
            min_y=y
            min_y_point=(x,y)
            min_index=count+index
        if count==count_limit and find_max:
            local_maximum.append(max_y_point)
            count=0
            index=max_index
            find_max=False
            min_y=max(input_points, key=lambda x: x[1])[1]
        elif count==count_limit and not find_max:
            local_minimum.append(min_y_point)
            count=0
            index=min_index
            find_max=True
            max_y=0
        count=count+1
    return local_maximum, local_minimum

def etch_between_points_in_mask(mask, point1, point2, thickness=5):
    # 确保输入的掩模是单通道的（例如，灰度图像）
    if len(mask.shape) != 2:
        raise ValueError("输入的掩模必须是二维的（单通道）")

    # 在掩模上绘制线条，从 point1 到 point2
    cv2.line(mask, point1, point2, 0, thickness=thickness)  # 设置线条颜色为 0（黑色）

    return mask

if __name__=='__main__':
    image_file_path='./data/pics/caries-0.8510638-272-735_1_2022021402.png'
    label_info, raw_mask, find_both_path_bool, file_name = get_info_from_data(image_file_path)
    # if not find_both_path_bool:
    #     continue        
    # yolov8_seg_label_list=[]
    # index_int_list=[]
    masks_index_dict={}
    dental_contour_combination=['Crown','Restoration', 'Dentin' ,'Enamel', 'Pulp', 'Root_canal_filling', 'Post_and_core', 'Caries']

    dental_contour=np.zeros_like(raw_mask).astype(np.uint8)
    # crown_points=[]
    # denti_points=[]
    for index_str, label in label_info.items():
        index_int = int(index_str)
        mask = (raw_mask == index_int).astype(np.uint8) * 255
        if label in dental_contour_combination:
            dental_contour=cv2.bitwise_or(dental_contour, mask)
        masks_index_dict[index_str]=mask

    upper_points = []
    lower_points = []
    #處理dental contour
    contours, _ = cv2.findContours(dental_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 獲取所有 y 值
        ys = contour[:, 0, 1]
        # 獲取上下邊界的 y 值
        upper_y = np.min(ys)
        lower_y = np.max(ys)

        # 將上半部和下半部分開（以上半部分的高度中點作為分界）
        #middle_y = (upper_y + lower_y) / 2
        middle_y=lower_y+(upper_y - lower_y)*0.73
        for point in contour:
            x, y = point[0]
            if y <= middle_y:
                upper_points.append([x, y])
            else:
                lower_points.append([x, y])
    # 將點轉換為 numpy 陣列

    lower_points.sort(key=lambda x: x[0])
    upper_points.sort(key=lambda x: x[0])
    filter_upper_points, _=find_the_extreme_points(upper_points, 100)
    _, filter_lower_points=find_the_extreme_points(lower_points, 400)

    #drawing_image = cv2.cvtColor(np.zeros_like(raw_mask).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    drawing_image = cv2.cvtColor(dental_contour, cv2.COLOR_GRAY2BGR)
    for point in filter_lower_points:
        cv2.circle(drawing_image, point, 5, (0, 0, 255), -1)
    for point in filter_upper_points:
        cv2.circle(drawing_image, point, 5, (255, 0, 0), -1)

    for lower_point in filter_lower_points:
        closest_upper_point = None  # 初始化最近的 upper_point
        min_distance = float('inf')  # 初始化最小距离为无穷大
        # 遍历每个 upper_point，寻找最近的一个
        for upper_point in filter_upper_points:
            distance = np.linalg.norm(np.array(lower_point) - np.array(upper_point))  # 计算距离
            if distance < min_distance:
                min_distance = distance
                closest_upper_point = upper_point
        # 如果找到最近的 upper_point，绘制连接线
        if closest_upper_point is not None:
            cv2.line(drawing_image, tuple(lower_point), tuple(closest_upper_point), (255, 0, 0), 2)  # 红色线条
        dental_contour=etch_between_points_in_mask(dental_contour, lower_point, closest_upper_point, thickness=20)
    show_two(drawing_image, dental_contour) 