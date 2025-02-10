
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import json
import yaml
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
# PIXEL_THRESHOLD = 2000  # 設定閾值，僅保留像素數大於該值的區域
# AREA_THRESHOLD = 500 # 設定閾值，避免過小的分割區域
# DISTANCE_THRESHOLD = 200 # 定義距離閾值（例如：設定 10 為最大可接受距離）
# SHORT_SIDE = 120 # 轉動短邊判斷閾值
# TWO_POINT_TEETH_THRESHOLD = 259 # 初判單雙牙尖使用
# RANGE_FOR_TOOTH_TIP_LEFT = 80 # 強迫判斷雙牙尖，中心區域定義使用(左)
# RANGE_FOR_TOOTH_TIP_RIGHT = 40 # 強迫判斷雙牙尖，中心區域定義使用(右)
with open('./conf/dental_measure_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)
for key, value in config.items():
    globals()[key] = value
def show_two(img1, img2, title1="Image 1", title2="Image 2"):
    """
    顯示兩張圖像並設定標題
    
    參數:
    - img1: 第一張圖像
    - img2: 第二張圖像
    - title1: 第一張圖像的標題
    - title2: 第二張圖像的標題
    """
    # 檢查圖像是否為灰階，若是則轉換成 RGB 以便 matplotlib 正確顯示
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # 使用 matplotlib 顯示兩張圖像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")  # 關閉坐標軸

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")  # 關閉坐標軸

    plt.show()
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
# ---------- 函式定義 ------------ #


def calculate_distance_with_scale(p1,p2,scale_x,scale_y):
    p1_array = np.array([p1[0]*scale_x, p1[1]*scale_y])
    p2_array = np.array([p2[0]*scale_x, p2[1]*scale_y])
    return np.linalg.norm(p1_array - p2_array)
    
# 根據 mask 判斷轉正使用的旋轉角度
def get_rotation_angle(mask):
    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    cnt = contours[0]

    # 根據輪廓取得方框
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]

    # 牙齒長邊在左右兩側，確保長邊是垂直於水平線
    if rect[1][0] > rect[1][1]:
        angle += 90
    if angle > 90:
        angle += 180
    return angle

# 利用 cv2 進行轉動
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 建立轉移角度用矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 利用建立的矩陣進行轉移轉動
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

# 把指定座標轉移回去原圖角度
def convert_coord_back(coord, angle, image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # 建立轉移回原角度用矩陣
    M_inv = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    
    # 因為只轉動單座標，該轉移矩陣要有另一項進行矩陣運算，這裡放單位矩陣做運算配合用
    coord_ones = np.array([coord[0], coord[1], 1.0])
    
    # 轉回原先角度
    original_coord_back = M_inv.dot(coord_ones)
    
    # 取得原圖座標，並且轉為整數
    original_coord_back = original_coord_back[:2].astype(int)
    
    return original_coord_back
# 檢查數值是否在指定範圍中
def is_within_range(value, target, range_size=50):
    return target - range_size <= value <= target + range_size

def assign_non_none_values(dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y):
    # 檢查 dentin_left_x 和 dentin_right_x 是否只找到其中一邊，是則判定為單牙尖
    if dentin_left_x is None and dentin_right_x is not None:
        dentin_left_x = dentin_right_x
    elif dentin_right_x is None and dentin_left_x is not None:
        dentin_right_x = dentin_left_x

    # 檢查 dentin_left_y 和 dentin_right_y 是否存在，是則判定為單牙尖
    if dentin_left_y is None and dentin_right_y is not None:
        dentin_left_y = dentin_right_y
    elif dentin_right_y is None and dentin_left_y is not None:
        dentin_right_y = dentin_left_y

    return dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y

# 根據指定順序，照點座標高度(y座標)排序
def get_top_points(contours, reverse=True):
    all_points = []
    # 取得輪廓中所有點，並且放到同一 list
    for contour in contours:
        sorted_points = sorted(contour, key=lambda x: x[0][1], reverse=reverse)
        top_points = sorted_points
        all_points.extend(top_points)
    # 排序
    all_points = sorted(all_points, key=lambda x: x[0][1], reverse=reverse)
    all_points = np.int32(all_points)
    return all_points



# ---------- 影像處理與遮罩相關函式 ------------ #

def load_images_and_masks(dir_path, target_dir):
    """載入影像及其對應的遮罩"""
    paths = {
        'gum': f"gum_{target_dir}.png",
        'teeth': f"teeth_{target_dir}.png",
        'dental_crown': f"dentalcrown_{target_dir}.png",
        'crown': f"crown_{target_dir}.png",
        'dentin': f"dentin_{target_dir}.png",
        'original': f"raw_{target_dir}.png"
    }
    images = {name: cv2.imread(os.path.join(dir_path, path), cv2.IMREAD_GRAYSCALE if name != 'original' else cv2.IMREAD_COLOR) 
              for name, path in paths.items()}
    return images

def threshold_images(images):
    """將影像轉為二值圖"""
    binary_images = {}
    for key, img in images.items():
        if key != 'original':
            _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            binary_images[key] = binary_img
    return binary_images

def clean_mask(mask, kernel_x=1, kernel_y=1, iterations=5):
    ###only odd number can allow for kneral size
    kernel_tuple=(2*kernel_x+1, 2*kernel_y+1)
    kernel = np.ones(kernel_tuple, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask


# def filter_large_components(mask, pixel_threshold):
#     """過濾掉像素數量小於閾值的區域"""
#     num_labels, labels = cv2.connectedComponents(mask)
#     label_counts = np.bincount(labels.flatten())
#     filtered_image = np.zeros_like(mask)
#     for label in range(1, num_labels):
#         if label_counts[label] > pixel_threshold:
#             filtered_image[labels == label] = 255
#     return filtered_image
# ---------- 影像分析與特徵提取相關函式 ------------ #
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

def find_brightest_pixel(image, x, y, bright_threshold=50, search_range=50):
    # 如果當前像素的亮度小於 threshold，就開始搜尋
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image[y, x] < bright_threshold:
        max_brightness = -1  # 記錄最亮像素的亮度
        brightest_pixel = (x, y)  # 記錄最亮像素的位置

        # 遍歷範圍內的像素，尋找亮度最高的像素
        for i in range(max(0, y - search_range), min(image.shape[0], y + search_range)):
            for j in range(max(0, x - search_range), min(image.shape[1], x + search_range)):
                if image[i, j] > max_brightness:
                    max_brightness = image[i, j]
                    brightest_pixel = (j, i)  # 更新最亮像素的位置

        return brightest_pixel

    # 如果當前像素本身足夠亮，返回該像素
    return (x, y)


def locate_points_with_dental_crown(image, dental_crown_bin, dilated_mask, mid_x, mid_y, overlay, crown_mask=None):

    """處理與 dental_crown 之交點 (Enamel的底端)"""
    # 獲取每個獨立 mask 與原始 mask 的交集區域
    crown_bool=False
    if dilated_mask is None:
        return None, None, None, None
    intersection=np.zeros_like(dilated_mask)
    ENAMEL_SKIP_PIXEL=int(ENAMEL_SKIP_PIXEL_RATIO*dilated_mask.shape[1])
    # 初始化取得座標，預設左右兩邊皆有
    enamel_left_x = enamel_left_y = enamel_right_x = enamel_right_y = 0
    leftmost = None
    rightmost = None
    area_ratio=np.inf
    #膨脹後的crown跟膨脹後的denti取交集
    if dental_crown_bin is not None:
        intersection = cv2.bitwise_and(dental_crown_bin, dilated_mask)
        area_ratio = cv2.countNonZero(intersection)/dilated_mask.shape[0]*dilated_mask.shape[1]

    #如果交集為0，則找尋跟假牙的交點 (如果假牙太近會誤判)
    if crown_mask is not None and area_ratio<ENAMEL_INTERSECTION_THRESHOLD_RATIO:
        intersection = cv2.bitwise_and(crown_mask, dilated_mask)
        area_ratio = cv2.countNonZero(intersection)/dilated_mask.shape[0]*dilated_mask.shape[1]
        if not np.all(area_ratio==0):
            crown_bool=True

    #如果真的再沒有就直接從dilated mask上面拿
    if dilated_mask is not None and area_ratio==0:
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        height = int(h * ENAMEL_NONE_CROWN_REPLACED_HEIGHT_RATIO)
        intersection[y:y+height, x:x+w] = dilated_mask[y:y+height, x:x+w]
        
    #show_plot(intersection)
    overlay[intersection > 0] = (255, 0, 0)  # 將 dentin 顯示
    # 取得交集區域的 contour 作為交點
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None
    # 將交點進行排序
    if crown_bool:#假牙邏輯
        for contour in contours:
            for point in contour:
                x, y = point[0]
                # 更新最左邊的座標
                if leftmost is None or x < leftmost[0]:
                    leftmost = (x, y)
                # 更新最右邊的座標
                if rightmost is None or x > rightmost[0]:
                    rightmost = (x, y)
        return leftmost[0], leftmost[1], rightmost[0], rightmost[1]

    #如果不是假牙就繼續
    corners = get_top_points(contours, reverse=True)
    if corners is None:
        return None, None, None, None 
    # # 針對每個點進行處理
    for corner in corners:
        x, y = corner.ravel()
        if x < mid_x and y > enamel_left_y and not is_within_range(x, enamel_left_x, ENAMEL_SKIP_PIXEL):
            enamel_left_x = x
            enamel_left_y = y
        elif x > mid_x and y > enamel_right_y and not is_within_range(x, enamel_right_x, ENAMEL_SKIP_PIXEL):
            enamel_right_x = x
            enamel_right_y = y

    if enamel_left_x!=0 and enamel_left_y!=0:
        enamel_left_x, enamel_left_y=find_brightest_pixel(image, enamel_left_x, enamel_left_y, bright_threshold=50, search_range=30)
    if enamel_right_x==0 and enamel_right_y==0:
        enamel_right_x, enamel_right_y=find_brightest_pixel(image, enamel_right_x, enamel_right_y, bright_threshold=50, search_range=30)

    #依舊是0的就返回none
    if enamel_left_x==0 and enamel_left_y==0:
        enamel_left_x = enamel_left_y = None
    if enamel_right_x==0 and enamel_right_y==0:
        enamel_right_x = enamel_right_y = None

    #breakpoint()
    return enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y

def locate_points_with_gum(gum_bin, dilated_mask, mid_x, mid_y, overlay):
    """處理與 gum 之交點 (Alveolar_bone的頂端)"""
    # 獲取每個獨立 mask 與原始 mask 的交集區域
    locate_gum_mid_x_threshold = gum_bin.shape[1]*GUM_LOCATE_GUM_MID_X_THRESHOLD_RATIO
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
            if x >= mid_x-locate_gum_mid_x_threshold and x <= mid_x+locate_gum_mid_x_threshold:
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
    
    height, width = image.shape[:2]
    max_length = max(width, height)
    DENTI_TWO_POINT_TEETH_THRESHOLD=max_length*DENTI_TWO_POINT_TEETH_THRESHOLD_RATIO
    DENTI_RANGE_FOR_TOOTH_TIP=max_length*DENTI_RANGE_FOR_TOOTH_TIP_RATIO
    #DENTI_RANGE_FOR_TOOTH_TIP_RIGHT=max_length*DENTI_RANGE_FOR_TOOTH_TIP_RIGHT_RATIO
    DENTI_RANGE_Y_LEFT_RIGHT_DENTIN=height*DENTI_RANGE_Y_LEFT_RIGHT_DENTIN_RATIO
    DENTI_RANGE_X_LEFT_RIGHT_DENTIN=width*DENTI_RANGE_X_LEFT_RIGHT_DENTIN_RATIO

    # 根據短邊大小(寬度)，初步判斷單牙尖或雙牙尖
    #print('short_side', short_side, TWO_POINT_TEETH_THRESHOLD)
    if short_side > DENTI_TWO_POINT_TEETH_THRESHOLD:
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
            if x < c_mid_x-DENTI_RANGE_FOR_TOOTH_TIP:
                left_corners.append(point)
            # 雙牙尖，太中間的點不可能是牙尖
            elif x >= c_mid_x-DENTI_RANGE_FOR_TOOTH_TIP and x <= c_mid_x+DENTI_RANGE_FOR_TOOTH_TIP:
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
        #print("Debugging ::")
        if all(v is None for v in [dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y]):
            print("All variables are None in locate_points_with_dentin.")
        else:
            if not is_within_range(dentin_left_y, dentin_right_y, DENTI_RANGE_Y_LEFT_RIGHT_DENTIN):
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
        if is_within_range(dentin_left_x, dentin_right_x, DENTI_RANGE_X_LEFT_RIGHT_DENTIN):
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
    if any([coord is None for coord in dentin_left_coord]) or any([coord is None for coord in dentin_left_coord]):
        return dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y
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


def extract_largest_component(mask):
    """提取遮罩中的最大連通區域"""
    num_labels, labels = cv2.connectedComponents(mask)
    label_counts = np.bincount(labels.flatten())
    max_label = np.argmax(label_counts[1:]) + 1
    largest_component = np.zeros_like(mask)
    largest_component[labels == max_label] = 255
    return largest_component

def combine_masks(binary_images):
    """合併所有遮罩，形成完整的合併遮罩"""
    combined_mask = None
    for key in ['dental_crown', 'dentin', 'binary_images']:
        if key in binary_images:
            combined_mask = cv2.bitwise_or(combined_mask, binary_images[key]) if combined_mask is not None else binary_images[key]
    return combined_mask

# def combine_masks(gum_mask, binary_images):
#     """合併所有遮罩，形成完整的合併遮罩"""
#     breakpoint()
#     combined_mask = cv2.bitwise_or(gum_mask, binary_images['teeth'])
#     combined_mask = cv2.bitwise_or(combined_mask, binary_images['dental_crown'])
#     combined_mask = cv2.bitwise_or(combined_mask, binary_images['dentin'])
#     return combined_mask
def draw_point(prediction, image_for_drawing):
    # 定義點的顏色
    points = {
        "enamel_left": (0, 0, 255),
        "enamel_right": (0, 0, 255),
        "gum_left": (0, 255, 0),
        "gum_right": (0, 255, 0),
        "dentin_left": (255, 0, 0),
        "dentin_right": (255, 0, 0)
    }
    
    # 印出所有點資訊
    for key, color in points.items():
        #print(f"{key} : ", prediction[key])
        cv2.circle(image_for_drawing, prediction[key], 5, color, -1)
    
    # 印出 Mid Points
    #print("Mid Points : ", prediction["mid"])
    
    return image_for_drawing

def draw_line(prediction, line_image, scale):
    #font size setting
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 1  # Font scale (size)
    thickness = 2  # Thickness of the text
    line_type = cv2.LINE_AA  # Anti-aliased line type for smooth text    
    
    height, width = line_image.shape[:2]
    #w_threshold = w / 10
    w_threshold = 20
    # if w<h:
    #     scale_w, scale_h= (31/w, 41/h)
    # else:
    #     scale_w, scale_h= (41/w, 31/h)
    scale_h, scale_w = scale
    # Define line pairs and their corresponding colors
    line_pairs = [
        (("enamel_left", "dentin_left", (0, 0, 255)), ("enamel_left", "gum_left", (0, 255, 255))),
        (("enamel_right", "dentin_right", (0, 0, 255)), ("enamel_right", "gum_right", (0, 255, 255)))
    ]

    def draw_line_and_text(line_image, p1, p2, color, text_position, show_text=True):
        cv2.line(line_image, p1, p2, color, 2)  # Draw enamel -> dentin line
        length=calculate_distance_with_scale(p1, p2, scale_h, scale_w)
        if show_text==True:
            cv2.putText(line_image, f'{length:.2f} mm', text_position, font, font_scale, color, thickness, line_type)
        return length
    def determine_stage_text(length, length2):
        stage_text="III"
        ratio=length/length2
        #print(ratio)
        if ratio < 0.15 and length < 2:
            stage_text = "0"
        elif ratio < 0.15:
            stage_text = "I"
        elif ratio <= 0.33:
            stage_text = "II"
        return stage_text 
    
    dental_pair_list=[]
    # Iterate through the line pairs and draw lines
    for i, ((start, end, color), (start2, end2, color2)) in enumerate(line_pairs):
        # Fetch the coordinates from the prediction
        p1, p2, p3, p4 = (prediction.get(start), prediction.get(end),
                          prediction.get(start2), prediction.get(end2))

        # print(f"{start} -> {end}: {p1}, {p2}")
        # print(f"{start2} -> {end2}: {p3}, {p4}")
        

        if any(pt is None or None in pt for pt in [p1, p2, p3, p4]):
            continue
        
        valid_point_check=all(pt is not None for pt in [p1, p2, p3, p4])
        #not_boundary_point_check=all(w_threshold < pt[0] < width - w_threshold for pt in [p1, p2, p3, p4])

        # Check if any of the points are None and skip this loop iteration if true
        #print(valid_point_check, not_boundary_point_check)

        # Check if all points are valid and within the threshold range

        if valid_point_check:
            if i%2==1:
                length2=draw_line_and_text(line_image, p3, p4, color2, text_position=p3, show_text=False) #CAL
                right_plot_text_point=(p4[0]-70,p4[1]+30)
                cv2.putText(line_image, f'{length2:.2f} mm', right_plot_text_point, font, font_scale, color2, thickness, line_type)
            else:
                length2=draw_line_and_text(line_image, p3, p4, color2, text_position=p3) #CAL
            length=draw_line_and_text(line_image, p1, p2, color, text_position=p2, show_text=False) #RL
            stage_text=determine_stage_text(length2, length)
            enamel_denti_mid_position=((p1[0]+p2[0])//2,(p1[1]+p2[1])//2)
            cv2.putText(line_image, stage_text, enamel_denti_mid_position, font, font_scale, (255, 255, 0), thickness, line_type)
            #breakpoint()
            dental_pair_list.append(
                {
                'side_id': i,
                'CEJ': (int(p1[0]),int(p1[1])),
                'ALC': (int(p4[0]),int(p4[1])),
                'APEX': (int(p2[0]),int(p2[1])),
                'TRL': float(length),
                'CAL': float(length2),
                'ABLD': float(length2/length),
                'stage': stage_text,
                }

        )


    return line_image, dental_pair_list

def int_processing(val):
    if val is None:
        return None  # 或者設為 0，根據需求決定
    return int(val)


def draw_image_and_print_information(prediction, image_for_drawing, line_image):
    # 繪圖及印出資訊
    # print("Mid Points : ", prediction["mid"])
    # print("enamel_left : ", prediction["enamel_left"])
    # print("enamel_right : ", prediction["enamel_right"])
    # print("gum_left : ", prediction["gum_left"])
    # print("gum_right : ", prediction["gum_right"])
    # print("dentin_left : ", prediction["dentin_left"])
    # print("dentin_right : ", prediction["dentin_right"])
    cv2.circle(image_for_drawing, prediction["enamel_left"], 5, (0, 0, 255), -1)
    cv2.circle(image_for_drawing, prediction["enamel_right"], 5, (0, 0, 255), -1)
    cv2.circle(image_for_drawing, prediction["gum_left"], 5, (0, 255, 0), -1)
    cv2.circle(image_for_drawing, prediction["gum_right"], 5, (0, 255, 0), -1)
    cv2.circle(image_for_drawing, prediction["dentin_left"], 5, (255, 0, 0), -1)
    cv2.circle(image_for_drawing, prediction["dentin_right"], 5, (255, 0, 0), -1)
    # Draw lines between points
    #print("e_l -> d_l : ", prediction["enamel_left"], prediction["dentin_left"])
    if (prediction["enamel_left"][0] is not None) and (prediction["dentin_left"] is not None):
        cv2.line(line_image, prediction["enamel_left"], prediction["dentin_left"], (0, 0, 255), 2)
    else:
        print("None Detected. Not drawing line.")
        
    #print("e_l -> g_l : ", prediction["enamel_left"], prediction["gum_left"])
    if (prediction["enamel_left"][0] is not None) and (prediction["gum_left"][0] is not None):
        cv2.line(line_image, prediction["enamel_left"], prediction["gum_left"], (0, 255, 255), 2)
    else:
        print("None Detected. Not drawing line.")
        
    #print("e_r -> d_r : ", prediction["enamel_right"], prediction["dentin_right"])
    if (prediction["enamel_right"][0] is not None) and (prediction["dentin_right"] is not None):
        cv2.line(line_image, prediction["enamel_right"], prediction["dentin_right"], (0, 0, 255), 2)
    else:
        print("None Detected. Not drawing line.")
    
    #print("e_r -> g_r : ", prediction["enamel_right"], prediction["gum_right"])
    if (prediction["enamel_right"][0] is not None) and (prediction["gum_right"][0] is not None):
        cv2.line(line_image, prediction["enamel_right"], prediction["gum_right"], (0, 255, 255), 2)
    else:
        print("None Detected. Not drawing line.")


# def process_and_save_predictions(predictions, dir_path, target_dir, correct_df):
#     """處理並儲存預測結果"""
#     sorted_predictions = sorted(predictions, key=lambda x: x['mid'][0])
#     df = pd.DataFrame(sorted_predictions)
    
#     if len(df) == 0:
#         df = correct_df.drop(index=df.index)
#         df.to_excel(os.path.join(dir_path, f"{target_dir}_comparison_results.xlsx"), index=False)
#         return

#     df = restructure_dataframe(df)
#     df_combined = combine_and_clean_dataframe(df)

#     # 儲存合併結果
#     df_cleaned = df_combined.dropna()
#     if len(df_cleaned) != 0:
#         df_cleaned['percentage'], df_cleaned['predicted_stage'] = zip(*df_cleaned.apply(calculate_predicted_stage, axis=1)) # 計算 stage
#     df_true_cleaned = prepare_true_dataframe(correct_df)

#     df_merged = merge_dataframes(df_cleaned, df_true_cleaned)
#     df_merged = df_merged.rename(columns={'牙齒ID（相對該張影像的順序ID即可、從左至右）':'tooth_id', 
#                         "牙尖ID（從左側至右側，看是連線到哪一個牙尖端）":"dentin_id",
#                         "珐瑯質跟象牙質交接點x":"enamel_x", "珐瑯質跟象牙質交接點y":"enamel_y",
#                         "牙齦交接點x":"gum_x" , "牙齦交接點y":"gum_y",
#                         "牙本體尖端點x":"dentin_x" , "牙本體尖端點y":"dentin_y" ,
#                         "長度":"length","stage":"true_stage"
#                         })
#     df_merged.to_excel(os.path.join(dir_path, f"{target_dir}_comparison_results.xlsx"), index=False)






def find_symmetry_axis(mask):
    # 计算质心
    moments = cv2.moments(mask)
    cX = int(moments['m10'] / moments['m00'])
    cY = int(moments['m01'] / moments['m00'])
    
    # 初步假设对称轴是垂直于质心的方向
    # 可以根据实际情况调整对称轴的选择方式
    return cX, cY

def map_point_to_symmetry_axis(cX, cY, point):
    # 将给定的点映射到对称轴的另一侧
    # 假设对称轴是垂直方向的，即 x = cX
    mapped_point = (2 * cX - point[0], point[1])
    return mapped_point

def find_closest_contour_point(contours, point):
    # 计算给定点和每个轮廓点的距离，找到最近的一个
    min_dist = float('inf')
    closest_point = None
    for contour in contours:
        for p in contour:
            dist = np.linalg.norm(np.array(p[0]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_point = p[0]
    return closest_point


def find_symmetry_axis(mask):
    # 找到mask的所有非零點
    points = np.column_stack(np.where(mask > 0))
    
    # 使用 PCA 找到主方向
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # 主軸方向
    direction = pca.components_[0]
    center = pca.mean_
    
    # 計算對稱軸（過中心並垂直於主軸方向）
    normal = np.array([-direction[1], direction[0]])  # 垂直方向
    # Plot mask with symmetry axis
    # plt.imshow(mask, cmap='gray')
    # plt.scatter(points[:, 1], points[:, 0], marker='.', color='blue', alpha=0.3)
    # plt.arrow(center[1], center[0], direction[1] * 50, direction[0] * 50, color='red', head_width=5)
    # plt.title("Symmetry Axis")
    # plt.show()    
    return (center[1],center[0]), (normal[1],normal[0])

def reflect_point(p, center, normal):
    # 計算點 p 關於對稱軸的鏡像點 p'
    p = np.array(p)
    center = np.array(center)
    normal = np.array(normal)
    
    v = p - center
    d = np.dot(v, normal)
    p_prime = p - 2 * d * normal

    # plt.scatter([p[1], p_prime[1]], [p[0], p_prime[0]], color=['green', 'red'])
    # plt.plot([p[1], p_prime[1]], [p[0], p_prime[0]], 'r--')
    # plt.title("Point Reflection")
    # plt.show()

    return p_prime

def find_edge_points(contour, image_shape, threshold):
    edge_points = []
    
    for point in contour:
        if is_point_near_edge(point[0], image_shape, threshold):
            edge_points.append(point[0])
    
    return edge_points
def connect_points(edge_points):
    # 使用 convex hull 來連接邊界點
    edge_points = sorted(edge_points, key=lambda p: p[0], reverse=True)  # 按照 x排序
    edge_points = edge_points[:len(edge_points)//5]
    p1 = max(edge_points, key=lambda p: p[1])  # y 最大
    p2 = min(edge_points, key=lambda p: p[1])  # y 最小 
    num_points=20
    x_values = np.linspace(p1[0], p2[0], num_points + 2)  # 包含兩端點
    y_values = np.linspace(p1[1], p2[1], num_points + 2)  # 包含兩端點
    return np.array(list(zip(x_values[1:-1], y_values[1:-1])))  # 返回中間的點    
    # if len(edge_points) > 2:
    #     hull = cv2.convexHull(np.array(edge_points))
    #     return hull
    #return np.array(edge_points)

def find_closest_contour_point(mask, p_prime):
    # def pad_mask(mask, pad=60):
    #     return cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)    
    # # 找到輪廓點
    # mask = pad_mask(mask)
    #mask = cv2.inpaint(mask, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #add_edges_to_contour(contours[0], mask.shape, mask.shape/10)
    edge_points = find_edge_points(contours[0], mask.shape[:2], mask.shape[0]/10)
    contour_points = np.vstack(contours[0])
    if len(edge_points)>=5:
        connected_edge_points = connect_points(edge_points).reshape(-1, 2)    
        contour_points= np.concatenate([contour_points,connected_edge_points],axis=0)
    # 使用 KDTree 找最近點
    tree = KDTree(contour_points)
    _, idx = tree.query(p_prime)
    closest_point = contour_points[idx]

    # Plot contour and closest point
    # plt.imshow(mask, cmap='gray')
    # plt.scatter(contour_points[:, 0], contour_points[:, 1], marker='.', color='blue', alpha=0.5)
    # plt.scatter(p_prime[0], p_prime[1], color='red', label='Reflected Point')
    # plt.scatter(closest_point[0], closest_point[1], color='yellow', label='Closest Contour Point')
    # plt.legend()
    # plt.title("Closest Contour Point")
    # plt.show()
    #breakpoint()
    return closest_point

def is_point_near_edge(point, image_shape, threshold):
    """
    檢查一個點是否靠近圖片邊緣。

    :param point: tuple，點的坐標 (x, y)
    :param image_shape: tuple，圖片的形狀 (height, width)
    :param threshold: int，定義靠近邊緣的距離閾值
    :return: bool，True 如果點靠近邊緣，否則 False
    """
    height, width = image_shape
    x, y = point
    
    # 檢查點距離四個邊的距離
    distance_to_left = x
    distance_to_right = width - x
    distance_to_top = y
    distance_to_bottom = height - y
    
    # 判斷是否有任何一個距離小於閾值
    if (distance_to_left < threshold or
        distance_to_right < threshold or
        distance_to_top < threshold or
        distance_to_bottom < threshold):
        return True
    
    return False