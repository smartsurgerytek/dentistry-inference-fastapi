import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# def split_erase():
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from src.allocation.domain.dental_measure.main import get_mask_dict_from_model
from src.allocation.domain.dental_measure.utils import *
from ultralytics import YOLO

Side_Shift = 3
Side_Contou_THD = 100
Match_Shift = 10
Match_Dist = 50
Window_Group = 100
Slide_Group = 4

def get_end_point(mask_sobj_in, flag_dir = 0):
    
    y_arr, x_arr = np.nonzero(mask_sobj_in)
    
    sorted_indx = np.argsort(y_arr)
    
    if(flag_dir == 0):#Top
        return x_arr[sorted_indx[-1]], y_arr[sorted_indx[-1]]
        
    else:#Bottom
        return x_arr[sorted_indx[0]], y_arr[sorted_indx[0]]

def mask_divide(pair_rect_x, mask_inter_in, rdentin_bin):
    
    mask_divide = np.copy(rdentin_bin)
    
    i = 0
    while(i < len(pair_rect_x)):
        st = pair_rect_x[i]
        
        # Copy and clear outoff rect.
        mask_tmp = np.copy(mask_inter_in)
        mask_tmp[:, 0:st] = 0
        mask_tmp[:, st + Window_Group:] = 0
        
        coordinates = np.argwhere(mask_tmp == 255)
#         coord_list = [tuple(coord) for coord in coordinates]
        
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(coord_list)
        
#         print(kmeans.cluster_centers_)
#         print(type(kmeans.cluster_centers_))

        # start_point = (int(kmeans.cluster_centers_[0][1]), int(kmeans.cluster_centers_[0][0]))
        # end_point = (int(kmeans.cluster_centers_[1][1]), int(kmeans.cluster_centers_[1][0]))
        
        
        
        # 使用 DBSCAN 進行聚類
        dbscan = DBSCAN(eps=15, min_samples=10)  # 調整 eps 和 min_samples 根據數據密度
        dbscan_labels = dbscan.fit_predict(coordinates)

        # 確保有至少兩個有效簇
        unique_labels = set(dbscan_labels) - {-1}  # 排除噪聲 (-1 為噪聲標籤)
        if len(unique_labels) < 2:
            print("DBSCAN 未能找到兩個有效的簇，請調整 eps 或 min_samples")
            return mask_divide

        # 計算每個簇的中心點
        cluster_centers = []
        cluster_centers_x = []
        for label in unique_labels:
            cluster_points = coordinates[dbscan_labels == label]
            center = np.mean(cluster_points, axis=0)  # 計算簇中心
            cluster_centers.append(center)
            cluster_centers_x.append(center[0])

        # 確保有兩個簇中心點
        if len(cluster_centers) > 2:
             print("聚類結果 > 2")

        # sort_indx = np.argsort(cluster_centers_x)  
            
        # 將中心點轉換為整數
        start_point = (int(cluster_centers[0][1]), int(cluster_centers[0][0]))
        end_point = (int(cluster_centers[1][1]), int(cluster_centers[1][0]))

        print(f"Start Point: {start_point}, End Point: {end_point}")
        
        cv2.line(mask_divide, start_point, end_point, 0, 4)
        cv2.line(rdentin_bin, start_point, end_point, 0, 4)
    
        i += 1
    
    return mask_divide
    
def get_pair_rect(mask_inter_in, mask_org):
    
    mask_inter_cp = np.copy(mask_inter_in)
    group_inter = np.zeros_like(mask_inter_cp)
    width = mask_inter_cp.shape[1]
    
    st_x = 0
    ed_x = st_x + Window_Group
    list_can_x = [] # candidator
    list_can_area = [] #candidator
    find_flag = 0 # If find one pair
    pair_rect_st_x = []
    while (ed_x < width):
        mk_win = mask_inter_cp[:, st_x:ed_x]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mk_win, connectivity=8)
        
        print(num_labels)
        
        if num_labels <= 2:
            if find_flag == 1:
                find_flag = 0
               
                sort_indx = np.argsort(list_can_area)
                can_st_x = list_can_x[sort_indx[-1]]
                pair_rect_st_x.append(can_st_x)
                
                #Clear intersections for avoiding repeat rect.
                mask_inter_cp[:, can_st_x:can_st_x+Window_Group] = 0
                
                list_can_x.clear()
                list_can_area.clear()
            
            
            st_x += Slide_Group
            ed_x = st_x + Window_Group    
                    
            continue
        
        else:
            find_flag = 1

            tot_area = 0
            for i in range(1, num_labels):
                tot_area += stats[i, cv2.CC_STAT_AREA]        
            list_can_area.append(tot_area)
            list_can_x.append(st_x)

            
            st_x += Slide_Group
            ed_x = st_x + Window_Group

            # print('find ' + str(st_x))
        
        
    print('find a pair rect. st. x = {}'.format(pair_rect_st_x))
    
    
    return pair_rect_st_x
    
    
    
def mask_intersec(mask_lt_in, mask_rt_in, rec_y, rec_h, comp_mask_in):
    mk_lt = np.copy(mask_lt_in)
    mk_rt = np.copy(mask_rt_in)
    comp_mask = np.copy(comp_mask_in)
    
    # Use WarpAffine for x axis shift
    height, width = mk_lt.shape[:2]
    M = np.float32([[1, 0, Match_Shift], [0, 1, 0]])
    mk_lt_sf = cv2.warpAffine(mk_lt, M, (width, height))
    M = np.float32([[1, 0, -Match_Shift], [0, 1, 0]])
    mk_rt_sf = cv2.warpAffine(mk_rt, M, (width, height))
    
    mask_inter = cv2.bitwise_and(mk_lt_sf, mk_rt_sf)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inter, connectivity=8)
    intersec_id = 1
    for i in range(1, num_labels):
        if centroids[i][1] < rec_y + 0.5*rec_h:
                continue
        
    return mask_inter
    
    
#Filter Noise Components
def process_mask(mask_in):
    mask = np.zeros_like(mask_in)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_in, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > Side_Contou_THD :
            mask[labels == i] = 255
            print(stats[i, cv2.CC_STAT_AREA])

    # cv2.imshow('img-mask-in2', mask)
            
    return mask
            
def enhance_split_detin(rdentin_bin):
    
    num_labels, labels = cv2.connectedComponents(rdentin_bin)

    for i in range(1, num_labels):
    
        component_mask = np.uint8(labels == i) * 255
        mask_rgb = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2RGB)
        
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
        
        cv2.drawContours(mask_rgb, [boxp], 0, (0,255,0), 3)
        # cv2.imshow('mask_rgb', mask_rgb)

        
        mask_inter = mask_intersec(diff_mask2, diff_mask1, rec_y, rec_h, component_mask)
        
        
        comb_mask_or = cv2.bitwise_or(diff_mask1, diff_mask2)
#         comb_mask_and = cv2.bitwise_and(diff_mask1, diff_mask2)
        
        pair_rect_x = get_pair_rect(mask_inter, component_mask)
        
        mk_divide = mask_divide(pair_rect_x, mask_inter, rdentin_bin)
    
        # cv2.imshow('img-mask-org', component_mask)
        # cv2.imshow('img-diff1', diff_mask1)
        # cv2.imshow('img-diff2', diff_mask2)
        # cv2.imshow('img-comb-or', comb_mask_or)
        # cv2.imshow('mask-intersec', mask_inter)
        # cv2.imshow('mask-divide', mk_divide)
        
        # cv2.waitKey(0)
        
    # cv2.destroyAllWindows()

def filling_holes(dental_contour):
    kernel = np.ones((10,10), np.uint8)  
    dental_contour = cv2.morphologyEx(dental_contour, cv2.MORPH_CLOSE, kernel)
    # 找到所有輪廓
    contours, _ = cv2.findContours(dental_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 建立一個空白圖像來繪製輪廓
    output_image = np.zeros_like(dental_contour)

    # 遍歷每個輪廓，檢查是否是孔洞，並填充它
    for contour in contours:
        # 計算輪廓的面積，如果面積小於某個閾值，則忽略它
        area = cv2.contourArea(contour)
        if area > 100:  # 可以根據需要調整面積閾值
            # 使用fillPoly填充每個輪廓內的區域
            cv2.fillPoly(output_image, [contour], 255)
    show_two(dental_contour, output_image)
    return output_image


if __name__ == "__main__":
    components_model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
    image = cv2.imread('./val_data/normal/nomal-x-ray-0.8510638-270-740_1_2022011013.png')
    components_model_masks_dict=get_mask_dict_from_model(components_model, image, method='semantic')
    dental_contour_combination=['Crown','Restoration', 'Dentin' ,'Enamel', 'Pulp', 'Root_canal_filling', 'Post_and_core', 'Caries']

    dental_contour=np.zeros((960,1280)).astype(np.uint8)
    # crown_points=[]
    # denti_points=[]
    for label_name, mask in components_model_masks_dict.items():
        if label_name in dental_contour_combination:
            dental_contour=cv2.bitwise_or(dental_contour, mask)
    #show_plot(dental_contour)

    dental_contour=filling_holes(dental_contour)

    enhance_split_detin(dental_contour)

    
