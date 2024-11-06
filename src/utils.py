
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
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
# 計算基於 ['enamel_x'] 和 ['enamel_y'] 的距離函數
def calculate_distance(row_true, row_cleaned):
    true_values = np.array([row_true['enamel_x'], row_true['enamel_y']])
    cleaned_values = np.array([row_cleaned['enamel_x_predicted'], row_cleaned['enamel_y_predicted']])
    return np.linalg.norm(true_values - cleaned_values)

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
    return all_points
               
# 計算 percentage 和期數，預測資料使用
def calculate_predicted_stage(row):
    enamel_x, enamel_y = row['enamel_x_predicted'], row['enamel_y_predicted']
    gum_x, gum_y = row['gum_x_predicted'], row['gum_y_predicted']
    dentin_x, dentin_y = row['dentin_x_predicted'], row['dentin_y_predicted']
    
    # 計算 A, B, C 點之間的距離
    AB = np.sqrt((enamel_x - gum_x) ** 2 + (enamel_y - gum_y) ** 2)
    AC = np.sqrt((enamel_x - dentin_x) ** 2 + (enamel_y - dentin_y) ** 2)
    
    # 計算 percentage
    percentage = (AB / AC) * 100
    
    # 判斷期數
    if percentage < 15:
        stage = "1"
    elif 15 <= percentage <= 33:
        stage = "2"
    else:
        stage = "3"
    
    return percentage, stage


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

def clean_mask(mask, kernel_size=(3, 3), iterations=5):
    """清理影像中的雜點"""
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask

def filter_large_components(mask, pixel_threshold):
    """過濾掉像素數量小於閾值的區域"""
    num_labels, labels = cv2.connectedComponents(mask)
    label_counts = np.bincount(labels.flatten())
    filtered_image = np.zeros_like(mask)
    for label in range(1, num_labels):
        if label_counts[label] > pixel_threshold:
            filtered_image[labels == label] = 255
    return filtered_image

# ---------- 影像分析與特徵提取相關函式 ------------ #

def extract_features(binary_images, original_img):
    """從遮罩中提取特徵點與區域資訊"""
    overlay = original_img.copy()
    line_image = original_img.copy()
    kernel = np.ones((3, 3), np.uint8)

    # 清理各個遮罩
    binary_images['dental_crown'] = clean_mask(binary_images['dental_crown'])
    binary_images['dentin'] = clean_mask(binary_images['dentin'], kernel_size=(30, 1), iterations=1)
    binary_images['gum'] = clean_mask(binary_images['gum'], kernel_size=(30, 1), iterations=2)

    # 保留最大區域
    binary_images['gum'] = extract_largest_component(binary_images['gum'])

    # 膨脹處理後的 gum
    binary_images['gum'] = cv2.dilate(binary_images['gum'], kernel, iterations=10)

    # 合併所有遮罩
    combined_mask = combine_masks(binary_images)
    non_masked_area = cv2.bitwise_not(combined_mask)

     # 繪製 overlay
    key_color_mapping={
        'dental_crown': (163, 118, 158),
        'dentin':(117, 122, 152),
        'gum': (0, 177, 177),
        'crown': (255, 0, 128),
    }
    for key in key_color_mapping.keys():
        if binary_images.get(key) is not None:
            overlay[binary_images[key] > 0] = key_color_mapping[key]
    # overlay[binary_images["dental_crown"] > 0] = (163, 118, 158)  # 將 dental_crown 顯示
    # overlay[binary_images["dentin"] > 0] = (117, 122, 152)  # 將 dentin 顯示
    # overlay[binary_images['gum'] > 0] = (0, 177, 177)  # 將 dentin 顯示
    # overlay[binary_images['crown'] > 0] = (255, 0, 128) # 將 crown 顯示

    # 回傳疊加後的影像和線條影像
    return overlay, line_image, non_masked_area

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


