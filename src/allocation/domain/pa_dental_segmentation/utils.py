import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened
    
def get_label_text_img(pred_index_labels, width, color_dict, text_dict):
    unique_list = np.unique(pred_index_labels).tolist()
    if 13 in unique_list:
        unique_list=[item for item in unique_list if item != 13] #pop out 13: background
    label_list_len = len(unique_list)
    col_num_maximum=3
    pos_idx = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    height = (label_list_len // 3) * 30 + 5
    
    visual_bias=50
    x_offset=(width-col_num_maximum*300)//2+visual_bias #offset is the setting offset for setting the text in the middle, 50 
    # height = (label_list_len // 2) * 30 + 5
    #height = 30+5
    #width = 1280
    
    label_img = np.zeros((height, width, 3), np.uint8)
    
    if label_list_len >= 1:
        #print("顏色輸出label_list_len")   #1117 表示有找到顏色輸出
        #print('label number', label_list_len)   #1117 表示有找到顏色輸出
        for i in range(label_list_len):
            label_num = unique_list[i]
            #print(label_num, text_dict[label_num]) #1117 表示有找到顏色輸出的代號 charley

            #     print(' this is not Caries no: {0} '.format(label_num)) # 帶數字編號
            #     self.caries_flag='false'

            row_idx, col_idx = pos_idx[i]
            rectangle_st = (x_offset + (col_idx-1)*300, 5 + (row_idx-1) * 30)
            rectangle_ed = (x_offset+70 + (col_idx-1)*300, 30 + (row_idx-1) * 30)
            text_pos = (rectangle_ed[0]+5, rectangle_ed[1]-5)
            #print(color_dict[label_num]) #1117 表示有找到顏色輸出charley 
            
            cv2.rectangle(label_img, rectangle_st, rectangle_ed, color_dict[text_dict[label_num]], -1)
            cv2.putText(label_img, text_dict[label_num], text_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255,255,255), 1, cv2.LINE_AA)
            #print(label_img) #1117
    return label_img


def mask_to_rle(mask: np.ndarray) -> list[int]:

    flattened = mask.T.flatten(order='F')

    rle = []
    last_val = 0
    count = 0

    for val in flattened:
        if val == last_val:
            count += 1
        else:
            rle.append(count)
            count = 1
            last_val = val
    rle.append(count)
    return rle

def find_center_mask(mask_binary):
    moments = cv2.moments(mask_binary)

    # 計算質心
    if moments['m00'] != 0:  # 確保掩膜不全為零
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = None, None  # 如果掩膜全為零
    return (cx,cy)

def smooth_mask(mask: np.ndarray, smoothing_factor: float = 5.0, points_interp: int = 500) -> np.ndarray:
    """
    對 binary mask 中所有輪廓進行 B-spline 曲線平滑處理，保留層級結構（例如內部洞不要填滿）
    :param mask: 二值 mask（0 和 255）
    :param smoothing_factor: spline 平滑程度，越大越平滑
    :param points_interp: 每個輪廓的插值點數
    :return: 平滑後的二值 mask
    """
    # 找所有輪廓與層級資訊
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return mask

    smoothed_mask = np.zeros_like(mask)

    # hierarchy[0]: [Next, Previous, First_Child, Parent]
    hierarchy = hierarchy[0]

    for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
        cnt = cnt.squeeze()

        if cnt.ndim != 2 or cnt.shape[0] < 5:
            continue

        x, y = cnt[:, 0], cnt[:, 1]

        try:
            # B-spline 平滑
            tck, u = splprep([x, y], s=smoothing_factor, per=True)
            u_new = np.linspace(0, 1, points_interp)
            x_new, y_new = splev(u_new, tck)
            smooth_contour = np.stack([x_new, y_new], axis=-1).astype(np.int32)

            # 填滿方向：外部輪廓填白（255），內部洞填黑（0）
            is_hole = hier[3] != -1
            color = 0 if is_hole else 255
            cv2.drawContours(smoothed_mask, [smooth_contour], -1, color, thickness=cv2.FILLED)

        except Exception as e:
            print(f"Failed to smooth contour: {e}")
            continue

    return smoothed_mask

def remove_small_regions(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    移除二值 mask 中小於 min_area 的小區域 (雜點)
    :param mask: 二值 mask (0和255)
    :param min_area: 保留的最小面積
    :return: 處理後的 mask
    """
    # 確保是單通道二值
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 尋找輪廓並保留大區域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned_mask, [cnt], -1, 255, cv2.FILLED)

    return cleaned_mask

def show_plot(image):
    #cv2.imshow("OpenCV Image", image)
    # 使用 matplotlib 绘制图形
    plt.figure()
    plt.imshow(image)
    plt.show()

def show_two(img1, img2, title1='Image 1', title2='Image 2', main_title='Comparison'):
    """
    Show two images side by side using matplotlib.

    Parameters:
    - img1: first image (numpy array)
    - img2: second image (numpy array)
    - title1: title for first image
    - title2: title for second image
    - main_title: overall figure title
    """
    plt.figure(figsize=(10,5))
    plt.suptitle(main_title, fontsize=16)

    plt.subplot(1, 2, 1)
    if img1.ndim == 2:  # grayscale
        plt.imshow(img1, cmap='gray')
    else:
        plt.imshow(img1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if img2.ndim == 2:  # grayscale
        plt.imshow(img2, cmap='gray')
    else:
        plt.imshow(img2)
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()