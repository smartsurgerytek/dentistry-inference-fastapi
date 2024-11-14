import cv2
import numpy as np
import os
from utils import *
from scipy.signal import savgol_filter
def remove_jump_points(points, threshold):
    # 需要保留的點
    filtered_points = []

    # 遍歷每個點，排除"跳點"
    for i in range(len(points)):
        # 第一個點和最後一個點的邊界處理
        if i == 0:
            if abs(points[i][1] - points[i + 1][1]) <= threshold:
                filtered_points.append(points[i])
        elif i == len(points) - 1:
            if abs(points[i][1] - points[i - 1][1]) <= threshold:
                filtered_points.append(points[i])
        else:
            # 中間的點，檢查與前後點的差異
            if abs(points[i][1] - points[i - 1][1]) <= threshold and abs(points[i][1] - points[i + 1][1]) <= threshold:
                filtered_points.append(points[i])

    return filtered_points

def smooth_points(points, window_length=10, polyorder=2):
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder).astype(np.int64)
    points_smooth = np.column_stack([x, y_smooth]).tolist()
    return points_smooth
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

# def find_local_extrema(points, n_neighbors=3, threshold=0.1):
#     local_minima = []
#     local_maxima = []
    
#     # 提取 x 和 y 坐标
#     x = np.array([point[0] for point in points])
#     y = np.array([point[1] for point in points])

#     # 遍历数据点，排除前后不足 n 个邻居的边界点
#     for i in range(n_neighbors, len(points) - n_neighbors):
#         # 选择当前点的左右 n 个邻居
#         left_neighbors_y = y[i - n_neighbors:i]
#         right_neighbors_y = y[i + 1:i + 1 + n_neighbors]
        
#         # 当前点的 y 值
#         current_value_y = y[i]
        
#         # 判断当前点是否为局部最小值或最大值
#         is_local_min = current_value_y < min(np.min(left_neighbors_y), np.min(right_neighbors_y)) and \
#                        (np.max(left_neighbors_y) - current_value_y > threshold) and \
#                        (np.max(right_neighbors_y) - current_value_y > threshold)
        
#         is_local_max = current_value_y > max(np.max(left_neighbors_y), np.max(right_neighbors_y)) and \
#                        (current_value_y - np.min(left_neighbors_y) > threshold) and \
#                        (current_value_y - np.min(right_neighbors_y) > threshold)
        
#         # 如果是局部最小值，添加到结果列表
#         if is_local_min:
#             local_minima.append(points[i])  # 使用 [x, y] 点
#         # 如果是局部最大值，添加到结果列表
#         if is_local_max:
#             local_maxima.append(points[i])  # 使用 [x, y] 点
    
#     return local_minima, local_maxima


# def find_local_extrema2(points, n_neighbors=3, k=2):
#     local_minima = []
#     local_maxima = []
    
#     # 提取 x 和 y 坐标
#     x = np.array([point[0] for point in points])
#     y = np.array([point[1] for point in points])
    
#     # 计算 y 值的标准差
#     y_std = np.std(y)
    
#     # 根据标准差计算阈值
#     threshold = k * y_std
    
#     # 遍历数据点，排除前后不足 n 个邻居的边界点
#     for i in range(n_neighbors, len(points) - n_neighbors):
#         # 选择当前点的左右 n 个邻居
#         left_neighbors_y = y[i - n_neighbors:i]
#         right_neighbors_y = y[i + 1:i + 1 + n_neighbors]
        
#         # 当前点的 y 值
#         current_value_y = y[i]
        
#         # 判断当前点是否为局部最小值或最大值
#         is_local_min = current_value_y < min(np.min(left_neighbors_y), np.min(right_neighbors_y)) and \
#                        (np.max(left_neighbors_y) - current_value_y > threshold) and \
#                        (np.max(right_neighbors_y) - current_value_y > threshold)
        
#         is_local_max = current_value_y > max(np.max(left_neighbors_y), np.max(right_neighbors_y)) and \
#                        (current_value_y - np.min(left_neighbors_y) > threshold) and \
#                        (current_value_y - np.min(right_neighbors_y) > threshold)
        
#         # 如果是局部最小值，添加到结果列表
#         if is_local_min:
#             local_minima.append(points[i])  # 使用 [x, y] 点
#         # 如果是局部最大值，添加到结果列表
#         if is_local_max:
#             local_maxima.append(points[i])  # 使用 [x, y] 点
    
#     return local_minima, local_maxima

def etch_between_points_in_mask(mask, point1, point2, thickness=5):
    # 确保输入的掩模是单通道的（例如，灰度图像）
    if len(mask.shape) != 2:
        raise ValueError("输入的掩模必须是二维的（单通道）")

    # 在掩模上绘制线条，从 point1 到 point2
    cv2.line(mask, point1, point2, 0, thickness=thickness)  # 设置线条颜色为 0（黑色）

    return mask

if __name__=='__main__':
    image_file_path='./data/pics/caries-0.8510638-272-735_1_2022021402.png'
    image=cv2.imread(image_file_path)
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


    #處理dental contour
    contours, _ = cv2.findContours(dental_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    upper_limit_ratio=0.73
    lower_limit_ratio=0.7
    
    points_dict_list=[]
    for contour_index, contour in enumerate(contours):
        upper_points = []
        lower_points = []        
        # 獲取所有 y 值
        ys = contour[:, 0, 1]
        # 獲取上下邊界的 y 值
        upper_y = np.min(ys)
        lower_y = np.max(ys)

        # 將上半部和下半部分開（以上半部分的高度中點作為分界）
        #middle_y = (upper_y + lower_y) / 2
        upper_limit=lower_y+(upper_y - lower_y)*upper_limit_ratio
        lower_limit=lower_y+(upper_y - lower_y)*lower_limit_ratio
        for point in contour:
            x, y = point[0]
            if y <= upper_limit:
                upper_points.append([x, y])
            #elif y>lower_limit_ratio:
            if y>lower_limit:
                lower_points.append([x, y])
        lower_points.sort(key=lambda x: x[0])
        upper_points.sort(key=lambda x: x[0])
        lower_points=remove_jump_points(lower_points, threshold=100)
        upper_points=remove_jump_points(upper_points, threshold=100)
        lower_points=smooth_points(lower_points, window_length=100, polyorder=2)
        upper_points=smooth_points(upper_points, window_length=10, polyorder=2)
        # _, lower_points_extremals=find_local_extrema(lower_points, n_neighbors=1, threshold=0.1)
        # upper_points_extremals, _=find_local_extrema(upper_points, n_neighbors=1, threshold=0.1)
        #_, lower_points_extremals=find_local_extrema2(lower_points, n_neighbors=1, k=0.01)
        #upper_points_extremals, _=find_local_extrema2(upper_points, n_neighbors=1, k=0.01)
        _, lower_points_extremals=find_the_extreme_points(lower_points, 100)
        upper_points_extremals, _=find_the_extreme_points(upper_points, 80)        
        points_dict_list.append({
            'index':contour_index,
            'upper_points':upper_points,
            'lower_points':lower_points,
            'lower_points_extremals':lower_points_extremals,
            'upper_points_extremals':upper_points_extremals,
        })
        
    lower_points=[points for points_dict in points_dict_list for points in points_dict['lower_points']]
    upper_points=[points for points_dict in points_dict_list for points in points_dict['upper_points']]
    lower_points_extremals=[points for points_dict in points_dict_list for points in points_dict['lower_points_extremals']]
    upper_points_extremals=[points for points_dict in points_dict_list for points in points_dict['upper_points_extremals']]
    #breakpoint()

    #image=draw_points_on_image(image, lower_points)
    #image=draw_points_on_image(image, upper_points)
    image=draw_points_on_image(image, lower_points_extremals, color=(0,255,0))
    image=draw_points_on_image(image, upper_points_extremals, color=(0,0,255))
    show_plot(image)
    # 將點轉換為 numpy 陣列
    breakpoint()

    # filter_upper_points, _=find_the_extreme_points(upper_points, 100)
    # _, filter_lower_points=find_the_extreme_points(lower_points, 400)

    # #drawing_image = cv2.cvtColor(np.zeros_like(raw_mask).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    # drawing_image = cv2.cvtColor(dental_contour, cv2.COLOR_GRAY2BGR)
    # for point in filter_lower_points:
    #     cv2.circle(drawing_image, point, 5, (0, 0, 255), -1)
    # for point in filter_upper_points:
    #     cv2.circle(drawing_image, point, 5, (255, 0, 0), -1)

    # for lower_point in filter_lower_points:
    #     closest_upper_point = None  # 初始化最近的 upper_point
    #     min_distance = float('inf')  # 初始化最小距离为无穷大
    #     # 遍历每个 upper_point，寻找最近的一个
    #     for upper_point in filter_upper_points:
    #         distance = np.linalg.norm(np.array(lower_point) - np.array(upper_point))  # 计算距离
    #         if distance < min_distance:
    #             min_distance = distance
    #             closest_upper_point = upper_point
    #     # 如果找到最近的 upper_point，绘制连接线
    #     if closest_upper_point is not None:
    #         cv2.line(drawing_image, tuple(lower_point), tuple(closest_upper_point), (255, 0, 0), 2)  # 红色线条
    #     dental_contour=etch_between_points_in_mask(dental_contour, lower_point, closest_upper_point, thickness=20)
    # show_two(drawing_image, dental_contour) 