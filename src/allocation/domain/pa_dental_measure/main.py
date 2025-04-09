import cv2
from ultralytics import YOLO
import numpy as np
from src.allocation.domain.pa_dental_measure.utils import *
import yaml
# components_model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
# contour_model=YOLO('./models/dentistryContour_yolov11n-seg_4.46.pt')

def extract_features(masks_dict, original_img, config=None):

    """從遮罩中提取特徵點與區域資訊"""
    overlay = original_img.copy()
    line_image = original_img.copy()

    # 清理各個遮罩
    if masks_dict.get('dental_crown') is not None:
        masks_dict['dental_crown'] = clean_mask(masks_dict['dental_crown'], kernel_x=DENTAL_CROWN_KERNAL_X, kernel_y=DENTAL_CROWN_KERNAL_Y , iterations=DENTAL_CROWN_ITERATION)
    
    masks_dict['dentin'] = clean_mask(masks_dict['dentin'], kernel_x=DENTI_KERNAL_X, kernel_y=DENTI_KERNAL_Y, iterations=DENTI_ITERATION)

    gum_kernel = np.ones((2*GUM_KERNAL_X+1, 2*GUM_KERNAL_Y+1), np.uint8)
    masks_dict['gum'] = cv2.dilate(masks_dict['gum'], gum_kernel, iterations=GUM_ITERATION)
    
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


def locate_points(image, component_mask, binary_images, idx, overlay, config=None):
    if config is not None:
        for key, value in config.items():
            globals()[key] = value
    # def less_than_area_threshold(component_mask, area_threshold):
    #     """根據指定面積大小，過濾過小的分割區域"""
    #     area = cv2.countNonZero(component_mask)
    #     # 去除掉過小的分割區域
    #     if area < area_threshold:
    #         return True
    #     return False

    
    prediction = {}
    # AREA_THRESHOLD=image.shape[0]*image.shape[1]*AREA_THRESHOLD_RATIO
    # if less_than_area_threshold(component_mask, AREA_THRESHOLD):
    #     return prediction
    # 以方框框住該 component_mask，整數化
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0]) # 最小區域長方形
    box = cv2.boxPoints(rect) # 取得長方形座標
    box = np.int32(box) # 整數化 
    width = rect[1][0]  # 寬度
    height = rect[1][1]  # 高度
    short_side = min(width, height)  # 短邊
    long_side = max(width, height)

    # if short_side < SHORT_SIDE:
    #    print('less than short side, return [] prediction')
    #    return prediction
    
    # 判斷旋轉角度
    angle = get_rotation_angle(component_mask)
    # 如果長短邊差距在 30 內 (接近正方形)，不轉動
    if is_within_range(short_side, long_side, ROTATION_ANGLE_THRESHOLD):
        angle = 0
        
    # 膨脹獨立 dentin 分割區域
    kernel = np.ones((2*DENTAL_CONTOUR_KERNAL_X+1, 2*DENTAL_CONTOUR_KERNAL_Y+1), np.uint8)
    dilated_mask = cv2.dilate(component_mask, kernel, iterations=DENTAL_CONTOUR_ITERATION)
 
    # 取得中點
    mid_y, mid_x = get_mid_point(image, dilated_mask, idx)
    

    ########### 處理與 dental_crown 之交點 (Enamel的底端) ###########
    # if binary_images.get('crown') is not None:
    #     binary_images["dental_crown"]=cv2.bitwise_or(binary_images["dental_crown"], binary_images["crown"])

    enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y = locate_points_with_dental_crown(image, binary_images.get("dental_crown"), dilated_mask, mid_x, mid_y, overlay, binary_images.get('crown'), config)

    ########### 處理與 gum 之交點 (Alveolar_bone的頂端) ########### 
    gum_left_x, gum_left_y, gum_right_x, gum_right_y = locate_points_with_gum(binary_images["gum"], dilated_mask, mid_x, mid_y, overlay, config)
    ########### 處理與 dentin 的底端 ########### 
    dentin_left_x, dentin_left_y, dentin_right_x, dentin_right_y = locate_points_with_dentin(binary_images["gum"], dilated_mask, mid_x, mid_y, angle, short_side, image, component_mask, config)

    
    prediction = {"mid": (mid_x, mid_y), 
                "enamel_left": (enamel_left_x, enamel_left_y), "enamel_right":(enamel_right_x, enamel_right_y),
                "gum_left":(gum_left_x, gum_left_y), "gum_right": (gum_right_x, gum_right_y),
                "dentin_left":(dentin_left_x, dentin_left_y), "dentin_right":(dentin_right_x, dentin_right_y),
                }
    
    # symmetry patched for missing side
    def fill_missing_side(missing_side, available_side, component_mask):
        if missing_side[0] is None:  # 如果某一侧缺少点
            center, normal = find_symmetry_axis(component_mask)  # 获取对称轴
            p_prime = reflect_point(available_side, center, normal)  # 对称反射
            closest_point = find_closest_contour_point(component_mask, p_prime)  # 找到最近的轮廓点
            missing_side = closest_point  # 填充缺失的点
        return missing_side    
    
    for side in ['left', 'right']:  # 遍历左右两个方向
        left_keys = [f"{comp}_left" for comp in ['enamel', 'gum', 'dentin']]
        right_keys = [f"{comp}_right" for comp in ['enamel', 'gum', 'dentin']]
        
        # 如果左边有缺失且右边有数据，填充左边
        if side == 'left' and all(prediction[key][0] is not None for key in right_keys) and any(prediction[key][0] is None for key in left_keys):
            for key in ['enamel', 'gum', 'dentin']:
                prediction[f"{key}_left"] = fill_missing_side(prediction[f"{key}_left"], prediction[f"{key}_right"], component_mask)
        
        # 如果右边有缺失且左边有数据，填充右边
        elif side == 'right' and all(prediction[key][0] is not None for key in left_keys) and any(prediction[key][0] is None for key in right_keys):
            for key in ['enamel', 'gum', 'dentin']:
                prediction[f"{key}_right"] = fill_missing_side(prediction[f"{key}_right"], prediction[f"{key}_left"], component_mask)         

    for key, (x, y) in prediction.items():
        # 對每一個座標進行 safe_int 處理
        prediction[key] = (int_processing(x), int_processing(y))

    return prediction

def get_mask_dict_from_model(model, image, method='semantic', mask_threshold=0.5):
    results=model.predict(image, verbose=False)
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
    box_x_list=[]
    for mask, box in zip(masks.data, boxes):
        class_id = int(box.cls)# Get class ID and confidence
        #confidence = float(box.conf)=
        class_name = class_names[class_id] # Get class name
        mask_np = mask.cpu().numpy() # Convert mask to numpy array and resize to match original image
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > mask_threshold).astype(np.uint8) * 255 # Convert mask to binary image

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
            if box.xyxy.type()=='torch.cuda.FloatTensor':
                box_x_list.append(int(box.xyxy[0][0].cpu().numpy()))
            else:
                box_x_list.append(int(box.xyxy[0][0].numpy()))
    if box_x_list:
        sorted_indices = sorted(range(len(box_x_list)), key=lambda i: box_x_list[i])
        # masks_dict['dental_contour'] = [masks_dict['dental_contour'][i] for i in sorted_indices] # alan mod

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

# def split_erase():
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit


Side_Shift = 3
Side_Contou_THD = 100
Match_Shift = 10
Match_Shift_Sec = 20
Match_Dist = 50

Window_Group_Max = 160
Window_Group = 100
Window_Margin = 10
Slide_Group = 4

OP_HWD = 30

DBSCAN_EPS = 15
DBSCAN_MIN_SAMP = 5

PIT_RADIUS = 5

DIVIDE_LINE_EXT = 8
DIVIDE_LINE_SLOPE_THD = 0.75

def get_end_point(mask_sobj_in, flag_dir = 0):
    
    y_arr, x_arr = np.nonzero(mask_sobj_in)
    
    sorted_indx = np.argsort(y_arr)
    
    if(flag_dir == 0):#Top
        return x_arr[sorted_indx[-1]], y_arr[sorted_indx[-1]]
        
    else:#Bottom
        return x_arr[sorted_indx[0]], y_arr[sorted_indx[0]]

def Refine_CoordBoundary(x_in, hf_wd, min_x, max_x):
    
    x_st = x_in - hf_wd
    if(x_st < min_x):
        x_st = min_x
    
    x_ed = x_in + hf_wd
    if(x_ed > max_x):
        x_ed = max_x
    
    return round(x_st), round(x_ed)
    
    
def extend_line(p1, p2, length=DIVIDE_LINE_EXT):
    # 計算方向向量
    direction = np.array(p2) - np.array(p1)
    
    # 計算單位方向向量
    unit_direction = direction / np.linalg.norm(direction)

    # 前延伸點 (p1 向反方向延伸)
    new_p1 = np.array(p1) - length * unit_direction

    # 後延伸點 (p2 向同方向延伸)
    new_p2 = np.array(p2) + length * unit_direction

    return tuple(np.int32(np.round(new_p1))), tuple(np.int32(np.round(new_p2)))

    
def mask_split_in_win(st, win_group, mask_inter_in, mask_divide_in, rdentin_bin):
    
    # print('st = {}, win_group = {}'.format(st, win_group))
    divide_state = 0 # 0 is no split ;  1 is split done
    
    # Copy and clear outoff rect.
    mask_divide = np.copy(mask_divide_in)
    mask_tmp = np.copy(mask_inter_in)
    mask_tmp[:, 0:st] = 0
    mask_tmp[:, st + win_group:] = 0
        
    coordinates = np.argwhere(mask_tmp == 255)

    # 使用 DBSCAN 進行聚類
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP)
    dbscan_labels = dbscan.fit_predict(coordinates)

    # 確保有至少兩個有效簇
    unique_labels = set(dbscan_labels) - {-1}  # 排除噪聲 (-1 為噪聲標籤)
    if len(unique_labels) < 2:
        # print("DBSCAN did not find > 1 Group")
        ttt = 1
    else:
        # 計算每個簇的中心點
        cluster_centers = []
        for label in unique_labels:
            if(label != -1):
                cluster_points = coordinates[dbscan_labels == label]
                center = np.mean(cluster_points, axis=0)  # 計算簇中心
                cluster_centers.append(center)

                
        tot_cent = len(cluster_centers)
        start_point = [0,0]
        end_point = [0,0]
        if tot_cent > 2:
            # print("聚類結果 > 2")
            # print(cluster_centers)
            
            cent_st_id = 0
            cent_ed_id = 1
            while(cent_ed_id < tot_cent):
                start_point = (int(round(cluster_centers[cent_st_id][1])), int(round(cluster_centers[cent_st_id][0])))
                end_point = (int(round(cluster_centers[cent_ed_id][1])), int(round(cluster_centers[cent_ed_id][0])))
                    
                dx = (start_point[0] - end_point[0])
                if abs(dx) < 1:
                    dx = 1
                m = float(start_point[1] - end_point[1]) / float(dx)

                if(abs(m) > DIVIDE_LINE_SLOPE_THD):
                    # print(f"Mask divide:: Start Point Org: {start_point}, End Point Org: {end_point}, M: {m}")
                    start_point, end_point = extend_line(start_point, end_point)
                    # print(f"Mask divide:: Start Point Divided: {start_point}, End Point Divided: {end_point}, M: {m}")
                    cv2.line(mask_divide, start_point, end_point, 0, 4)
                    cv2.line(rdentin_bin, start_point, end_point, 0, 4)
                    del cluster_centers[0]
                    del cluster_centers[0]
                    
                    divide_state = 1
                else:
                    # print(f"Start Point: {start_point}, End Point: {end_point}, M: {m}")
                    del cluster_centers[1]
                    
                tot_cent = len(cluster_centers)
                    
        else:
            start_point = (int(round(cluster_centers[0][1])), int(round(cluster_centers[0][0])))
            end_point = (int(round(cluster_centers[1][1])), int(round(cluster_centers[1][0])))
                
            dx = (start_point[0] - end_point[0])
            if abs(dx) < 1:
                dx = 1
            m = float(start_point[1] - end_point[1]) / float(dx)

            if(abs(m) > DIVIDE_LINE_SLOPE_THD):
                # print(f"Mask divide:: Start Point Org: {start_point}, End Point Org: {end_point}, M: {m}")
                start_point, end_point = extend_line(start_point, end_point)
                # print(f"Mask divide:: Start Point Divided: {start_point}, End Point Divided: {end_point}, M: {m}")
                cv2.line(mask_divide, start_point, end_point, 0, 4)
                cv2.line(rdentin_bin, start_point, end_point, 0, 4)
                
                divide_state = 1
  

    return mask_divide, divide_state
    
# Refine rect's start x and rect window size (avoid vertical line for x cut at pit circle)
def margin_refine(can_st_x, can_win, mask_inter_in):
    
    mk_wd = mask_inter_in.shape[1]
    # marg_box_hwd = 0
    comp_width = 0
    
#     st_lt = can_st_x - marg_box_hwd
#     if st_lt < 0:
#         st_lt = 0 
#     st_rt = can_st_x + marg_box_hwd
#     if st_rt >= mk_wd:
#         st_rt = mk_wd - 1
        
    
#     ed_lt = can_st_x + can_win - marg_box_hwd
#     if ed_lt < 0:
#         ed_lt = 0 
#     ed_rt = can_st_x + can_win + marg_box_hwd 
#     if ed_rt >= mk_wd:
#         ed_rt = mk_wd - 1
        
    can_ed_org = can_st_x + can_win
    
    margin_region = mask_inter_in[:, can_st_x]
    reg_val = np.sum(np.sum(margin_region))
    if(reg_val > 0):
        comp_width = Window_Margin
        if can_st_x - Window_Margin < 0:
            comp_width = can_st_x
        can_st_x -= comp_width # Enlarge left to refine
        
     
    margin_region = mask_inter_in[:, can_ed_org]
    reg_val = np.sum(np.sum(margin_region))
    if(reg_val > 0):
        can_win += comp_width + Window_Margin # Enlarge Right to refine
    
    
    return can_st_x, can_win
    
def mask_split_func(mask_inter_in, rdentin_bin):
    mask_inter_cp = np.copy(mask_inter_in)
    mask_divide = np.copy(rdentin_bin)
    group_inter = np.zeros_like(mask_inter_cp)
    width = mask_inter_cp.shape[1]
    
    win_group = Window_Group_Max # init. oberservation window
    st_x = 0 # rect win start x
    ed_x = st_x + win_group # rect win end x
    list_can_x = [] # candidator rect's start x
    list_can_area = [] #candidator
    list_can_win = [] # candidator
    find_flag = 0 # If find one pair
    num_lab_pre = 1 # 1, 2, 3, > 3
    pair_rect_st_x = []
    pair_rect_win = []
    while (ed_x < width):
        mk_win = mask_inter_cp[:, st_x:ed_x]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mk_win, connectivity=8)
        
        # print('st = {}, num_labels = {}, win_group = {}'.format(st_x, num_labels, win_group))
        
        if num_labels < num_lab_pre: # if key pit number decreased, then back to get pair pits to split mask
            if find_flag == 1:
                
                # Get the largest Pit Area in Candidators
                sort_indx = np.argsort(list_can_area)
                can_st_x = list_can_x[sort_indx[-1]]
                can_win = list_can_win[sort_indx[-1]]
                
                # Refine rect's start x and rect window size (avoid vertical line for x cut at pit circle)
                can_st_x, can_win = margin_refine(can_st_x, can_win, np.copy(mask_inter_cp))
                
                # Add final rect's start x and rect window size to the List # record
                pair_rect_st_x.append(can_st_x) # record
                pair_rect_win.append(can_win) # record
                
                
                # Do split mask at this rect. window
                mask_divide, divide_state = mask_split_in_win(can_st_x, can_win, np.copy(mask_inter_cp), mask_divide, rdentin_bin)
                
                # print('done divide & remove st = {}, ed = {}, win_group = {}, divide_state={}'.format(can_st_x, can_st_x+can_win, can_win, divide_state))
                
                if(divide_state == 1): #Clear pit info. mask (intersections) for avoiding error by repeating process.
                    mask_inter_cp[:, can_st_x:(can_st_x+can_win)] = 0
                
                # cv2.imshow('done divide & remove', mask_inter_cp) # gui
                # cv2.waitKey(0) # gui
            
                #Reset for Next
                list_can_win.clear()
                list_can_x.clear()
                list_can_area.clear()
                
                find_flag = 0
                win_group = Window_Group_Max # Use default Window
                num_lab_pre = 1
            else:
                # Keep searching
                test_line = 0
            
            st_x += Slide_Group
            ed_x = st_x + win_group    
                    
        elif num_labels <= 2: # Keep searching
            st_x += Slide_Group
            ed_x = st_x + win_group    
        
        elif num_labels == 3: # Find a pair pits (candidator)
            find_flag = 1

            tot_area = 0
            for i in range(1, num_labels):
                tot_area += stats[i, cv2.CC_STAT_AREA]        
            list_can_area.append(tot_area)
            list_can_x.append(st_x)
            list_can_win.append(win_group)
            
            st_x += Slide_Group
            ed_x = st_x + win_group

            # print('find ' + str(st_x))
            
        elif num_labels > 3:
            # Find a pair pits but more than two pits (candidator)
            if(win_group == Window_Group_Max): # Keep searching
                win_group = Window_Group
                ed_x = st_x + win_group
                # print('Shrink')
            else: # Find
                find_flag = 1
                
                tot_area = 0
                for i in range(1, num_labels):
                    tot_area += stats[i, cv2.CC_STAT_AREA]        
                list_can_area.append(tot_area)
                list_can_x.append(st_x)
                list_can_win.append(win_group)

                st_x += Slide_Group
                ed_x = st_x + win_group
            
        
        num_lab_pre = num_labels # update number 
        
    # print('find pair rect. st. x = {}'.format(pair_rect_st_x))
    # print('find pair rect window group = {}'.format(pair_rect_win))
    
    
    return pair_rect_st_x, pair_rect_win, mask_divide
    
    
    
def quadratic_func(x, a, b, c):
    
    return a*x**2 + b*x + c
    
    
    
# Mask Intersection to Pit Key Circle Points
def intersec_refine(mask_inter_in, comp_mask_in):
    mask_inter_refine = np.zeros_like(mask_inter_in)
    op_mask_show = np.zeros_like(mask_inter_in) # gui show
    op_mask_cont_show = np.zeros_like(mask_inter_in) # gui show
    inter_show = np.zeros_like(mask_inter_in) # gui show
    inter_show = cv2.cvtColor(inter_show, cv2.COLOR_GRAY2BGR) # gui show
    comp_mask = np.copy(comp_mask_in)
    comp_mask_show = np.copy(comp_mask_in) # gui show
    comp_mask_show = cv2.cvtColor(comp_mask_show, cv2.COLOR_GRAY2BGR) # gui show
    
    # Get original Intersection (y,x)
    coordinates = np.argwhere(mask_inter_in == 255) # (y,x) type
    
    if(len(coordinates) < 2):
        return mask_inter_refine
    
    # Use DBSCAN to Cluster Intersection
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP)  # 調整 eps 和 min_samples 根據數據密度
    dbscan_labels = dbscan.fit_predict(coordinates)

    unique_labels = set(dbscan_labels) - {-1}  # 排除噪聲 (-1 為噪聲標籤)
    cluster_centers = [] # (x,y) type
    if len(unique_labels) < 2:
        # print("intersec_refine:: DBSCAN did not find > 1 Group")
        return mask_inter_refine
    else:
        # Get each Cluster's center
        for label in unique_labels:
            if label != -1: # 排除噪聲
                cluster_points = coordinates[dbscan_labels == label]
                center = np.mean(cluster_points, axis=0)  # 計算簇中心
                # print('intersec_refine:: {}'.format(center[[1,0]]))
                cluster_centers.append(center[[1,0]])

                
    # Process each Cluster (Intersection Coarse Center)
    tot_cent = len(cluster_centers)
    cent_id = 0
    OP_HWD = 30 # Pit Local Windows Box
    while(cent_id < tot_cent):
        clu_x, clu_y = cluster_centers[cent_id]
        
        # print('intersec_refine:: intersection center (x,y) = {}'.format((clu_x, clu_y)))
        cv2.circle(inter_show, (np.int32(clu_x), np.int32(clu_y)), 3, (0, 0, 255), -1)
        cv2.circle(comp_mask_show, (np.int32(clu_x), np.int32(clu_y)), 3, (0, 0, 255), -1)
        
        # Refine Local Box's Boundary
        comp_ht, comp_wd = comp_mask.shape
        x_min, x_max = Refine_CoordBoundary(clu_x, OP_HWD, 0, comp_wd-1)
        y_min, y_max = Refine_CoordBoundary(clu_y, OP_HWD, 0, comp_ht-1)
        
        # Get Local Dentin Mask
        op_mask = np.zeros_like(comp_mask)
        op_mask[y_min:y_max, x_min:x_max] = cv2.bitwise_not(comp_mask[y_min:y_max, x_min:x_max])
        op_mask_convex_hull = np.copy(op_mask) # for convex hull computing
        
        op_mask_show = cv2.bitwise_or(op_mask_show, op_mask) # gui
        
        # Get Local Dentin Mask Edge
        kernel = np.ones((3, 3), np.uint8)
        op_mask_ref = cv2.erode(op_mask, kernel, iterations=1)
        op_mask -= op_mask_ref
        op_mask[y_min:y_min+3, :] = 0
        op_mask[y_max-3:y_max, :] = 0
        
        op_mask_cont_show = cv2.bitwise_or(op_mask_cont_show, op_mask) # gui
        
        # Get Local Dentin Mask Edge Coordinate
        coord_op = np.argwhere(op_mask == 255) # (y,x) type
        coord_op = coord_op[:, [1,0]] # (x,y) type
        coord_op_x = coord_op[:, 0]
        coord_op_y = coord_op[:, 1]

        # Perform curve fitting
        # poly_coeff = np.polyfit(coord_op_x, coord_op_y, 2)  # 拟合二次多项式
        # poly = np.poly1d(poly_coeff)  # 生成多项式函数
        poly_coeff, covariance = curve_fit(quadratic_func, coord_op_x, coord_op_y)
        
        # Get Local Mask (Edge) Shape Type Info.
        coef_a, coef_b, coef_c = poly_coeff
        # op_x_v = -coef_b / (2 * coef_a)  # 计算 y 方向的顶点
        # op_y_v = poly(op_x_v)
        # op_y_v = quadratic_func(op_x_v, coef_a, coef_b, coef_c)
        
        # Show Curve Fitting Results  # gui
        x_vals = np.arange(x_min, x_max, 1)  # 取所有 x 范围
        # y_vals = poly(x_vals)
        y_vals = quadratic_func(x_vals, coef_a, coef_b, coef_c)
        for x, y in zip(x_vals, y_vals):
            cv2.circle(inter_show, (np.int32(x), np.int32(y)), 1, (0, 255, 255), -1)  # gui
        
        
        # Get Point Cloud Direction
        mean_coord = np.mean(coord_op, axis=0)
        coord_centered = coord_op - mean_coord
        cov_matrix = np.cov(coord_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Project Point Cloud to short axis
        coord_short_ax = np.abs(coord_centered @ eigenvectors[:, 1])
        
        # Determin Direction
        med_num = 5
        coord_sort_idx = np.argsort(coord_short_ax)
        coord_y_dist_min = np.median(coord_op[coord_sort_idx[0:med_num], 1])
        coord_y_dist_max = np.median(coord_op[coord_sort_idx[-med_num:], 1])
        
        coef_a = 1
        if(coord_y_dist_min > coord_y_dist_max):
            coef_a = -1
        
        # Refine Pit (x,y) Method:
        
        op_x_v = -1 # Pit Center x
        op_y_v = -1 # Pit Center y
        
        # Refine Method 1
        # Find Contours in the Local Dentin Mask
        contours, _ = cv2.findContours(op_mask_convex_hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the convex hull of the largest contour
        # Assuming the largest contour corresponds to the object of interest
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  # Get the largest contour
            convex_hull = cv2.convexHull(largest_contour)  # Compute the convex hull
            sorted_hull = convex_hull[convex_hull[:, 0, 1].argsort()]
            
            # Use Mask (Edge) Shape Type to Refine Top or Bottom Pit Center
            op_x_v_hull, op_y_v_hull = sorted_hull[0, 0, :]
            if(coef_a < 0): # If parabolic notch upward
                op_x_v_hull, op_y_v_hull = sorted_hull[-1, 0, :]
        
            # print('intersec_refine:: hull fitting refined pit (x,y) = {}'.format((op_x_v_hull, op_y_v_hull)))
            cv2.circle(inter_show, (np.int32(op_x_v_hull), np.int32(op_y_v_hull)), 3, (0, 255, 0), -1)
            cv2.circle(comp_mask_show, (np.int32(op_x_v_hull), np.int32(op_y_v_hull)), 3, (0, 255, 0), -1)
        
            op_x_v = op_x_v_hull
            op_y_v = op_y_v_hull
        
    
        # Refine Method 2 (Spare Method)
        # Use Mask (Edge) Shape Type to Get Top or Bottom Pit Center in cloud points
        sorted_coords = coord_op[coord_op[:, 1].argsort()]
        op_x_v_ed, op_y_v_ed = sorted_coords[0]
        if(coef_a < 0): # If parabolic notch upward
            op_x_v_ed, op_y_v_ed = sorted_coords[-1]
        
        # print('intersec_refine:: egde refined pit (x,y) = {}'.format((op_x_v_ed, op_y_v_ed)))
        cv2.circle(inter_show, (np.int32(op_x_v_ed), np.int32(op_y_v_ed)), 3, (255, 0, 0), -1)
        cv2.circle(comp_mask_show, (np.int32(op_x_v_ed), np.int32(op_y_v_ed)), 3, (255, 0, 0), -1)
        
        # If convex hull fitting is failed then use Refine Method 2
        if(op_x_v == -1 or op_y_v == -1):
            op_x_v = op_x_v_ed
            op_y_v = op_y_v_ed
        
        
        # Refined Pit Key Center on Mask Results (center circle size is a key parameter, too)
        cv2.circle(mask_inter_refine, (np.int32(op_x_v), np.int32(op_y_v)), PIT_RADIUS, 255, -1)
        
        cent_id += 1
    
    
    # cv2.imshow('intersec_refine:: op_mask_show', op_mask_show) # gui
    # cv2.imshow('intersec_refine:: op_mask_cont_show', op_mask_cont_show) # gui
    # cv2.imshow('intersec_refine:: inter_show', inter_show) # gui
    # cv2.imshow('intersec_refine:: comp_mask_show', comp_mask_show) # gui
    # cv2.waitKey(0)
    
    return mask_inter_refine
    
    
def mask_intersec(mask_lt_in, mask_rt_in, rec_y, rec_h, comp_mask_in):
    mk_lt = np.copy(mask_lt_in)
    mk_rt = np.copy(mask_rt_in)
    comp_mask = np.copy(comp_mask_in)
    
    # Shift Side Edge Mask
    height, width = mk_lt.shape[:2]
    M = np.float32([[1, 0, Match_Shift], [0, 1, 0]])
    mk_lt_sf = cv2.warpAffine(mk_lt, M, (width, height))
    M = np.float32([[1, 0, -Match_Shift], [0, 1, 0]])
    mk_rt_sf = cv2.warpAffine(mk_rt, M, (width, height))
    
    # Fast Intersection first
    mask_inter = cv2.bitwise_and(mk_lt_sf, mk_rt_sf)
    
    
    # Shift Side Edge Mask more
    height, width = mk_lt.shape[:2]
    M = np.float32([[1, 0, Match_Shift_Sec], [0, 1, 0]])
    mk_lt_sf = cv2.warpAffine(mk_lt, M, (width, height))
    M = np.float32([[1, 0, -Match_Shift_Sec], [0, 1, 0]])
    mk_rt_sf = cv2.warpAffine(mk_rt, M, (width, height))
    
    # Fast Intersection second
    mask_inter_sec = cv2.bitwise_and(mk_lt_sf, mk_rt_sf)
        
    # Get Each Intersection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inter, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 5:
            mask_inter[labels == i] = 0
            # print(stats[i, cv2.CC_STAT_AREA])
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inter, connectivity=8)
    
    num_labels_sec, labels_sec, stats_sec, centroids_sec = cv2.connectedComponentsWithStats(mask_inter_sec, connectivity=8)
    for i in range(1, num_labels_sec):
        if stats_sec[i, cv2.CC_STAT_AREA] < 5:
            mask_inter_sec[labels_sec == i] = 0
            # print(stats_sec[i, cv2.CC_STAT_AREA])
    
    num_labels_sec, labels_sec, stats_sec, centroids_sec = cv2.connectedComponentsWithStats(mask_inter_sec, connectivity=8)
    
    # Second Aux Intersection to Enhance Mask Intersection
    sec_idx = 1
    win_roi = 120
    while(sec_idx < num_labels_sec):
        x_sec, y_sec = centroids_sec[sec_idx]
        mask_roi = np.copy(mask_inter[int(y_sec - win_roi/2) : int(y_sec + win_roi/2), 
                                      int(x_sec - win_roi/4) : int(x_sec + win_roi/4)])
        if(np.sum(np.sum(mask_roi)) == 0):
            mask_inter[labels_sec == sec_idx] = 255
            # print('add')
        
        sec_idx += 1
    
    # cv2.imshow('org mask intersec', mask_inter) # gui show
    # cv2.waitKey(0)
    
    # Mask Intersection to Intersection Key Circle Points
    mask_inter = intersec_refine(mask_inter, comp_mask_in)
    
    return mask_inter
    
    
#Filter Noise Components
def process_mask(mask_in):
    mask = np.zeros_like(mask_in)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_in, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > Side_Contou_THD :
            mask[labels == i] = 255
            # print(stats[i, cv2.CC_STAT_AREA])

    # cv2.imshow('img-mask-in2', mask)
            
    return mask
            
# Enhace splitting Detin Mask 
def enhance_split_detin(rdentin_bin):
    num_labels, labels = cv2.connectedComponents(rdentin_bin)
    for i in range(1, num_labels):
        # Get each original detin mask
        component_mask = np.uint8(labels == i) * 255
        mask_rgb = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2RGB)
        
        # Get Key Side Edge (for Fast Key Intersection Pit)
        height, width = component_mask.shape[:2]
        # Define Translation Matrix
        M = np.float32([[1, 0, Side_Shift], [0, 1, 0]])
        # Use WarpAffine for x axis shift
        shifted_mask = cv2.warpAffine(component_mask, M, (width, height))
        diff_mask1 = np.int16(component_mask) - np.int16(shifted_mask)
        diff_mask1[diff_mask1 < 0] = 0
        diff_mask1 = np.uint8(diff_mask1)
        diff_mask1 = process_mask(diff_mask1)
        
        # Define Translation Matrix
        M = np.float32([[1, 0, -Side_Shift], [0, 1, 0]])
        # Use WarpAffine for x axis shift
        shifted_mask = cv2.warpAffine(component_mask, M, (width, height))
        diff_mask2 = np.int16(component_mask) - np.int16(shifted_mask)
        diff_mask2[diff_mask2 < 0] = 0
        diff_mask2 = np.uint8(diff_mask2)
        diff_mask2 = process_mask(diff_mask2)
        
        #Dilate to Enhance Feature
        kernel = np.ones((3, 3), np.uint8)
        diff_mask1 = cv2.dilate(diff_mask1, kernel, iterations=2)
        diff_mask2 = cv2.dilate(diff_mask2, kernel, iterations=2)

        # Get and Draw Min Rect.
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contou = contours[0]
        rec_x, rec_y, rec_w, rec_h = cv2.boundingRect(contou)
        rect = cv2.minAreaRect(contou)
        boxp = cv2.boxPoints(rect)
        boxp = np.int0(boxp)
        # print(boxp)
        
        # cv2.drawContours(mask_rgb, [boxp], 0, (0,255,0), 3)
        # cv2.imshow('mask_rgb', mask_rgb)

        # Combine mask to Show
        comb_mask_show = cv2.bitwise_or(diff_mask1, diff_mask2)
        
        # Fast Key Intersection Pit
        mask_inter = mask_intersec(diff_mask2, diff_mask1, rec_y, rec_h, component_mask)
        
        # Use Key Intersection Pit to Split Mask
        pair_rect_x, pair_rect_window, mk_divide = mask_split_func(mask_inter, rdentin_bin)
    
        # cv2.imshow('img-mask-org', component_mask)
        # cv2.imshow('img-diff1', diff_mask1)
        # cv2.imshow('img-diff2', diff_mask2)
        # cv2.imshow('comb_mask_show', comb_mask_show) # gui show
        # cv2.imshow('mask-intersec', mask_inter)
        # cv2.imshow('mask-divide', mk_divide)
        
        # cv2.waitKey(0)
        
    # cv2.imshow('dentin_bin-divide', rdentin_bin)
    # cv2.waitKey(0)
        
    # cv2.destroyAllWindows()

def dental_estimation(image, component_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='dict', config=None):
    if config is None:
        with open('./conf/best_dental_measure_parameters.yaml', 'r') as file:
            config = yaml.safe_load(file)
    if config is not None:
        for label, values in config.items():
            globals()[label] = values

    scale=(scale_x,scale_y)
    components_model_masks_dict=get_mask_dict_from_model(component_model, image, method='semantic', mask_threshold=DENTAL_MODEL_THRESHOLD)
    contours_model_masks_dict=get_mask_dict_from_model(contour_model, image, method='instance', mask_threshold=DENTAL_CONTOUR_MODEL_THRESHOLD)

    error_messages=''
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
        'gum': "No gum detected"
    }

    # check 'dentin' and 'gum' existed
    for component, error_message in required_components.items():
        if components_model_masks_dict.get(component) is None:
            error_messages=error_message
            return (generate_error_image(error_messages), error_messages) if 'image' in return_type else []
    # check 'dental_crown', 'crown' existed
    if (components_model_masks_dict.get('dental_crown') is None and components_model_masks_dict.get('crown') is None):
        error_messages="No dental_crown detected"
        return (generate_error_image(error_messages), error_messages) if 'image' in return_type else [] 
    
    # contour model check
    if contours_model_masks_dict.get('dental_contour') is None:
        error_messages = "No dental instance detected"
        return (generate_error_image(error_messages), error_messages) if 'image' in return_type else []
        
    # Retrive the dental_contour from contour_model

    components_model_masks_dict['dental_contour']=np.zeros(image.shape[:2], dtype=np.uint8)
    for dental_contour in contours_model_masks_dict['dental_contour']:
        components_model_masks_dict['dental_contour'] = cv2.bitwise_or(components_model_masks_dict['dental_contour'], dental_contour)
    
    # # Retrive the dental_contour from components_model
    # contour_elements=['dentin','Caries','dental_crown','crown','Implant','Restoration','Pulp','Post_and_core']
    # contours_from_component_model=np.zeros(image.shape[:2], dtype=np.uint8)
    # for key in contour_elements:
    #     if components_model_masks_dict.get(key) is not None:
    #         contours_from_component_model = cv2.bitwise_or(contours_from_component_model, components_model_masks_dict[key])
    # # combine two model contours
    # components_model_masks_dict['dental_contour']=cv2.bitwise_or(components_model_masks_dict['dental_contour'], contours_from_component_model)

    #retrive the crown or enamal mask
    crown_or_enamal_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in ['dental_crown', 'crown']:
        mask = components_model_masks_dict.get(label)
        if mask is not None:
            crown_or_enamal_mask = cv2.bitwise_or(crown_or_enamal_mask, mask)
        
    # retrive the dentin mask (dental_contours- crown_or_enamal_mask)
    denti_from_contour=components_model_masks_dict['dental_contour']-cv2.bitwise_and(components_model_masks_dict['dental_contour'], crown_or_enamal_mask)
    components_model_masks_dict['dentin']=denti_from_contour
    ## retrive the dentin mask (dental_contours- crown_or_enamal_mask) or (denti_from_contour)
    # components_model_masks_dict['dentin']=cv2.bitwise_or(components_model_masks_dict['dentin'], denti_from_contour)
    # if components_model_masks_dict.get('Pulp') is not None:
    #     components_model_masks_dict['dentin']=cv2.bitwise_or(components_model_masks_dict['dentin'], components_model_masks_dict['Pulp'])
    
    # clean crown mask
    if components_model_masks_dict.get('crown') is not None and components_model_masks_dict.get('dental_crown') is not None:
        components_model_masks_dict["dental_crown"]=components_model_masks_dict["dental_crown"]-cv2.bitwise_and(components_model_masks_dict["dental_crown"], components_model_masks_dict["crown"])

    overlay, line_image, non_masked_area= extract_features(components_model_masks_dict, image) # 處理繪圖用圖片等特徵處理後圖片

    

    predictions = []
    image_for_drawing=image.copy()
    #for i in range(1, num_labels):  # 從1開始，0是背景
    enhance_split_detin(components_model_masks_dict['dentin']) # alan mod
    num_labels, labels = cv2.connectedComponents(components_model_masks_dict['dentin'])
    
    #for i, component_mask in enumerate(contours_model_masks_dict['dental_contour']):
    for i in range(1, num_labels):
        component_mask = np.uint8(labels == i) * 255

        # 取得分析後的點
        prediction = locate_points(image_for_drawing, component_mask, components_model_masks_dict, i+1, overlay, config)
        # 如果無法判斷點，會回傳空字典
        if len(prediction) == 0:
            continue
        prediction["teeth_id"] = i
        prediction["dentin_id"] = None
        #print(f"Tooth {i}")
        image_for_drawing=draw_point(prediction, image_for_drawing)
        image_for_drawing, dental_pair_list=draw_line(prediction, image_for_drawing, scale)
        prediction['pair_measurements']=dental_pair_list
        prediction['teeth_center']=prediction['mid']
        if dental_pair_list:
            predictions.append(prediction)

    if return_type=='cvat':
        if not predictions:
            return []
        cvat_results=[]
        points_label=['CEJ','APEX','ALC']
        polyline_label=['CAL','TRL']
        tag_label=['ABLD','stage']
        polyline_mapping={
            'CAL': ['enamel','gum'],
            'TRL': ['enamel','dentin']
        }
        left_right=['left','right']
        for prediction in predictions:
            teeth_id=prediction['teeth_id']
            for pair_measurement in prediction['pair_measurements']:
                side_id=pair_measurement['side_id']
                for label, values in pair_measurement.items():
                    if label in points_label:
                        cvat_results.append({'label':label,
                                            'type':'point',
                                            'points':list(values), 
                                            'teeth_id':teeth_id, 
                                            'side_id':side_id})
                    elif label in polyline_label:
                        side=left_right[side_id]
                        enamel_key=polyline_mapping[label][0]+"_"+side
                        gum_or_dentin_key=polyline_mapping[label][1]+"_"+side
                        points=list(prediction[enamel_key]+prediction[gum_or_dentin_key])                    
                        cvat_results.append({'label':label,
                                            'type':'polyline',
                                            'points':points, 
                                            'attributes':[{
                                                'name':'length',
                                                'input_type':'number',
                                                'value': values,
                                            }],
                                            'teeth_id':teeth_id, 
                                            'side_id':side_id})
                meta_data_atrributes=[{'name': key,
                                        'input_type': 'number',
                                        'value': pair_measurement[key],} for key in tag_label]
                cvat_results.append({'label':'metadata',
                                    'type':'tag',
                                    'attributes': meta_data_atrributes,
                                    'teeth_id':teeth_id, 
                                    'side_id':side_id})     
        return cvat_results           

    if return_type=='image_array':
        return image_for_drawing, error_messages
    # elif return_type=='image_base64':
    #     return numpy_to_base64(image_for_drawing, image_format='PNG'), error_messages
    else:
        return predictions