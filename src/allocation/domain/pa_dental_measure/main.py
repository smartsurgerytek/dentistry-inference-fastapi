import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import cv2
from ultralytics import YOLO
import numpy as np
from src.allocation.domain.pa_dental_measure.utils import *
import yaml
# components_model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
# contour_model=YOLO('./models/dentistryContour_yolov11n-seg_4.46.pt')

def perform_morphology(masks_dict, original_img, config=None):

    """Extract feature points and region information from the mask"""
    overlay = original_img.copy()
    line_image = original_img.copy()

    # 清理各個遮罩
    if masks_dict.get('dental_crown') is not None:
        masks_dict['dental_crown'] = opening_mask(masks_dict['dental_crown'], kernel_x=DENTAL_CROWN_KERNAL_X, kernel_y=DENTAL_CROWN_KERNAL_Y , iterations=DENTAL_CROWN_ITERATION)
    
    masks_dict['dentin'] = opening_mask(masks_dict['dentin'], kernel_x=DENTI_KERNAL_X, kernel_y=DENTI_KERNAL_Y, iterations=DENTI_ITERATION)

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
        'artificial_crown': (255, 0, 128),
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

    enamel_left_x, enamel_left_y, enamel_right_x, enamel_right_y = locate_points_with_dental_crown(image, binary_images.get("dental_crown"), dilated_mask, mid_x, mid_y, overlay, binary_images.get('artificial_crown'), config)

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

    # get class name
    class_names = result.names
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs

    if masks is None:
        return {}
    
    masks_dict={}
    box_x_list=[]
    for mask, box in zip(masks.data, boxes):
        class_id = int(box.cls)# Get class ID and confidence
        class_name = class_names[class_id] # Get class name
        mask_np = mask.cpu().numpy() # Convert mask to numpy array and resize to match original image
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        mask_binary = (mask_np > mask_threshold).astype(np.uint8) * 255 # Convert mask to binary image

        if np.sum(mask_binary) == 0: #error handling
            continue

        if method=='semantic':
            masks_dict[class_name] = cv2.bitwise_or(masks_dict.get(class_name, 0), mask_binary)
        else:
            masks_dict.setdefault(class_name, []).append(mask_binary)
            box_x_list.append(int(box.xyxy[0][0].detach().cpu().numpy()))
            
    # dental_contour_model: this code is used for resorting mask order so that it is easy to oberserve the performance
    if box_x_list and masks_dict.get('dental_contour') is not None:
        sorted_indices = sorted(range(len(box_x_list)), key=lambda i: box_x_list[i])
        masks_dict['dental_contour'] = [masks_dict['dental_contour'][i] for i in sorted_indices] # alan mod
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


            
# Enhace splitting Detin Mask 
def enhance_split_detin(rdentin_bin, config):
    if config is not None:
        for label, values in config.items():
            globals()[label] = values

    num_labels, labels = cv2.connectedComponents(rdentin_bin)
    
    for i in range(1, num_labels):
        # Get each original detin mask
        component_mask = np.uint8(labels == i) * 255

        # Get Key Side Edge (for Fast Key Intersection Pit)
        height, width = component_mask.shape[:2]

        # Define Translation Matrix
        M = np.float32([[1, 0, DENTAL_SPLIT_SIDE_SHIFT], [0, 1, 0]])
        # Use WarpAffine for x axis shift
        shifted_mask = cv2.warpAffine(component_mask, M, (width, height))
        diff_mask1 = np.int16(component_mask) - np.int16(shifted_mask)
        diff_mask1[diff_mask1 < 0] = 0
        diff_mask1 = np.uint8(diff_mask1)
        diff_mask1 = select_main_dental_areas(diff_mask1, DENTAL_SPLIT_SIDE_CONTOUR_THRESHOLD)
        
        # Define Translation Matrix
        M = np.float32([[1, 0, -DENTAL_SPLIT_SIDE_SHIFT], [0, 1, 0]])

        # Use WarpAffine for x axis shift
        shifted_mask = cv2.warpAffine(component_mask, M, (width, height))
        diff_mask2 = np.int16(component_mask) - np.int16(shifted_mask)
        diff_mask2[diff_mask2 < 0] = 0
        diff_mask2 = np.uint8(diff_mask2)
        diff_mask2 = select_main_dental_areas(diff_mask2, DENTAL_SPLIT_SIDE_CONTOUR_THRESHOLD)
        
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
        boxp = np.intp(boxp)

        # Fast Key Intersection Pit
        mask_inter = mask_intersec(diff_mask2, 
                                   diff_mask1, 
                                   component_mask,
                                   config,
                                   )
        
        # Use Key Intersection Pit to Split Mask
        pair_rect_x, pair_rect_window, mk_divide = mask_split_func(mask_inter, rdentin_bin, config)

    return mk_divide
        

def update_with_contour_and_crown_info(components_model_masks_dict, contours_model_masks_dict, image):
    
    updated_masks_dict = components_model_masks_dict.copy() #make the function pure

    # Retrive the dental_contour from contour_model and add in components_model_masks_dict
    updated_masks_dict['dental_contour']=np.zeros(image.shape[:2], dtype=np.uint8)
    for dental_contour in contours_model_masks_dict['dental_contour']:
        updated_masks_dict['dental_contour'] = cv2.bitwise_or(updated_masks_dict['dental_contour'], dental_contour)

    ###Retrive the dental_contour from components_model 
    # contour_elements = ['dentin', 'Caries', 'dental_crown', 'artificial_crown', 'Implant', 'Restoration', 'Pulp', 'Post_and_core']
    # contours_from_component_model = np.zeros(image.shape[:2], dtype=np.uint8)
    # for mask in [updated_masks_dict[key] for key in contour_elements if updated_masks_dict.get(key) is not None]:
    #     contours_from_component_model = cv2.bitwise_or(contours_from_component_model, mask)

    ###combine two model contours (Note that: sometimes you will need to combine two way to consturct the mask)
    #updated_masks_dict['dental_contour']=cv2.bitwise_or(updated_masks_dict['dental_contour'], contours_from_component_model)

    #retrive the crown or enamal mask
    crown_or_enamal_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in ['dental_crown', 'artificial_crown']:
        mask = updated_masks_dict.get(label)
        if mask is not None:
            crown_or_enamal_mask = cv2.bitwise_or(crown_or_enamal_mask, mask)
        
    # retrive the dentin mask (dental_contours- crown_or_enamal_mask)
    denti_from_contour_model=updated_masks_dict['dental_contour']-cv2.bitwise_and(updated_masks_dict['dental_contour'], crown_or_enamal_mask)
    updated_masks_dict['dentin']=denti_from_contour_model
    ### retrive the dentin mask (dental_contours- crown_or_enamal_mask) or (denti_from_contour)
    # updated_masks_dict['dentin']=cv2.bitwise_or(updated_masks_dict['dentin'], denti_from_contour_model)
    # if updated_masks_dict.get('Pulp') is not None:
    #     updated_masks_dict['dentin']=cv2.bitwise_or(updated_masks_dict['dentin'], updated_masks_dict['Pulp'])
    
    # clean crown mask : the dental_crown sometimes have over lapped with the crown (the artifial material)
    if updated_masks_dict.get('artificial_crown') is not None and updated_masks_dict.get('dental_crown') is not None:
        updated_masks_dict["dental_crown"]=updated_masks_dict["dental_crown"]-cv2.bitwise_and(updated_masks_dict["dental_crown"], updated_masks_dict["artificial_crown"])

    return updated_masks_dict
def dental_estimation(image, component_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='dict', config=None):
    #read the config and assign const
    if config is None:
        with open('./conf/best_dental_measure_parameters.yaml', 'r') as file:
            config = yaml.safe_load(file)
    if config is not None:
        for label, values in config.items():
            globals()[label] = values

    error_messages=''
    # note that crown is a tooth-shaped "cap" made of artifial material that is placed over a damaged, decayed, or weakened tooth.
    # while the dental crown refers to the visible portion of a tooth that is above the gum line in atonomy.
    denti_measure_names_map={
        'Alveolar_bone': 'gum',
        'Dentin': 'dentin',
        'Enamel': 'dental_crown',
        'Crown': 'artificial_crown' ### In fact it is enamel (why labeling man so stupid)
    }
    required_components = {
        'dentin': "No dental instance detected",
        'gum': "No gum detected"
    }
    scale=(scale_x,scale_y)

    # get the model dict
    components_model_masks_dict_init=get_mask_dict_from_model(component_model,
                                                              image, 
                                                              method='semantic', 
                                                              mask_threshold=DENTAL_MODEL_THRESHOLD)
    components_model_masks_dict = {denti_measure_names_map.get(k, k): v for k, v in components_model_masks_dict_init.items()} #mapping the key

    contours_model_masks_dict=get_mask_dict_from_model(contour_model,
                                                       image, 
                                                       method='instance', 
                                                       mask_threshold=DENTAL_CONTOUR_MODEL_THRESHOLD)


    # check 'dentin' and 'gum' existed
    for component, error_message in required_components.items():
        if components_model_masks_dict.get(component) is None:
            error_messages=error_message
            return (generate_error_image(error_messages), error_messages) if 'image' in return_type else []
        
    # check 'dental_crown', 'crown' existed
    if all(components_model_masks_dict.get(key) is None for key in ['dental_crown', 'artificial_crown']):
        error_messages="No dental_crown detected"
        return (generate_error_image(error_messages), error_messages) if 'image' in return_type else [] 
    
    # contour model check
    if contours_model_masks_dict.get('dental_contour') is None:
        error_messages = "No dental instance detected"
        return (generate_error_image(error_messages), error_messages) if 'image' in return_type else []
        

    masks_dict=update_with_contour_and_crown_info(components_model_masks_dict, contours_model_masks_dict, image)

    overlay, line_image, non_masked_area= perform_morphology(masks_dict, image, config)

    predictions = []
    image_for_drawing=image.copy()
    #for i in range(1, num_labels):  # 從1開始，0是背景
    dentin_mask_splited=enhance_split_detin(masks_dict['dentin'], config) # alan mod

    num_labels, index_masks = cv2.connectedComponents(dentin_mask_splited)

    #for i, component_mask in enumerate(contours_model_masks_dict['dental_contour']):
    for i in range(1, num_labels):
        component_mask = np.uint8(index_masks == i) * 255

        # 取得分析後的點
        prediction = locate_points(image_for_drawing, component_mask, masks_dict, i+1, overlay, config)
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
    if not predictions:
        return []
    
    if return_type=='cvat':

        points_label = ['CEJ', 'APEX', 'ALC']
        polyline_label = ['CAL', 'TRL']
        tag_label = ['ABLD', 'stage']
        polyline_mapping = {
            'CAL': ['enamel', 'gum'],
            'TRL': ['enamel', 'dentin']
        }
        left_right = ['left', 'right']

        def append_point(prediction, label, values, teeth_id, side_id):
            return {
                    'label': label,
                    'type': 'point',
                    'points': list(values),
                    'teeth_id': teeth_id,
                    'side_id': side_id
                }

        def append_polyline(prediction, label, values, teeth_id, side_id):
            return {
                    'label': label,
                    'type': 'polyline',
                    'points': list(prediction[polyline_mapping[label][0] + "_" + left_right[side_id]] + 
                                prediction[polyline_mapping[label][1] + "_" + left_right[side_id]]),
                    'attributes': [{'name': 'length', 'input_type': 'number', 'value': values}],
                    'teeth_id': teeth_id,
                    'side_id': side_id
                }

        def append_metadata(pair_measurement, tag_label, teeth_id, side_id):
            return {
                    'label': 'metadata',
                    'type': 'tag',
                    'attributes': [{'name': key, 'input_type': 'number', 'value': pair_measurement[key]} for key in tag_label],
                    'teeth_id': teeth_id,
                    'side_id': side_id
                }

        cvat_results = []
        for prediction in predictions:
            teeth_id = prediction['teeth_id']
            for pair_measurement in prediction['pair_measurements']:
                side_id = pair_measurement['side_id']
                for label, values in pair_measurement.items():
                    if label in points_label:
                        cvat_results.append(append_point(prediction, label, values, teeth_id, side_id))
                    elif label in polyline_label:
                        cvat_results.append(append_polyline(prediction, label, values, teeth_id, side_id))

                cvat_results.append(append_metadata(pair_measurement, tag_label, teeth_id, side_id))
        
        # cvat_results=[]
        # points_label=['CEJ','APEX','ALC']
        # polyline_label=['CAL','TRL']
        # tag_label=['ABLD','stage']
        # polyline_mapping={
        #     'CAL': ['enamel','gum'],
        #     'TRL': ['enamel','dentin']
        # }
        # left_right=['left','right']
        # for prediction in predictions:
        #     teeth_id=prediction['teeth_id']
        #     for pair_measurement in prediction['pair_measurements']:
        #         side_id=pair_measurement['side_id']
        #         for label, values in pair_measurement.items():
        #             if label in points_label:
        #                 cvat_results.append({'label':label,
        #                                     'type':'point',
        #                                     'points':list(values), 
        #                                     'teeth_id':teeth_id, 
        #                                     'side_id':side_id})
        #             elif label in polyline_label:
        #                 side=left_right[side_id]
        #                 enamel_key=polyline_mapping[label][0]+"_"+side
        #                 gum_or_dentin_key=polyline_mapping[label][1]+"_"+side
        #                 points=list(prediction[enamel_key]+prediction[gum_or_dentin_key])                    
        #                 cvat_results.append({'label':label,
        #                                     'type':'polyline',
        #                                     'points':points, 
        #                                     'attributes':[{
        #                                         'name':'length',
        #                                         'input_type':'number',
        #                                         'value': values,
        #                                     }],
        #                                     'teeth_id':teeth_id, 
        #                                     'side_id':side_id})
        #         meta_data_atrributes=[{'name': key,
        #                                 'input_type': 'number',
        #                                 'value': pair_measurement[key],} for key in tag_label]
        #         cvat_results.append({'label':'metadata',
        #                             'type':'tag',
        #                             'attributes': meta_data_atrributes,
        #                             'teeth_id':teeth_id, 
        #                             'side_id':side_id})     
        return cvat_results           

    if return_type=='image_array':
        return image_for_drawing, error_messages
    # elif return_type=='image_base64':
    #     return numpy_to_base64(image_for_drawing, image_format='PNG'), error_messages
    else:
        return predictions
    

if __name__ == "__main__":
    image=cv2.imread("./tests/files/caries-0.6741573-260-760_1_2022052768.png")
    components_model=YOLO('./models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt')
    contour_model=YOLO('./models/dentistry_pa-contour_yolov11n-seg_24.46.pt')
    predictions=dental_estimation(image, components_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='dict', config=None)
    print(predictions)