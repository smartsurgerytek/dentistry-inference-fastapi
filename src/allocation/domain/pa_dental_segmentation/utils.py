import matplotlib.pyplot as plt
import numpy as np
import cv2

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