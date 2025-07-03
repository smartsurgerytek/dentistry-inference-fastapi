from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.measure import approximate_polygon
import os 
import sys
import yaml
from PIL import Image, ImageDraw, ImageFont
# from scipy.optimize import linprog
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from src.allocation.domain.pa_dental_segmentation.utils import *

def get_yolov8_label(mask_binary,tolerance=0.5):
    points = []
    # contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # #pick up the bigest area        
    # contour = max(contours, key=cv2.contourArea)

    # x_norm = [point[0][0] / mask_binary.shape[1] for point in contour]
    # y_norm = [point[0][1] / mask_binary.shape[0] for point in contour]
    # # Check if any coordinate exceeds the image dimensions
    # if any(x > 1 or y > 1 for x, y in zip(x_norm, y_norm)):
    #     assert False, "warning: contour coordinates exceed the image dimensions"
    # points.extend(list(zip(x_norm, y_norm)))
    contours = find_contours(mask_binary, level=0.5)
    
    for contour in contours:
        # Reduce number of points while maintaining shape accuracy
        simplified_contour = approximate_polygon(contour, tolerance=tolerance)
        # Convert to LabelMe format (x,y coordinates)
        x_norm = [float(point[0] / mask_binary.shape[0]) for point in simplified_contour]
        y_norm = [float(point[1] / mask_binary.shape[1]) for point in simplified_contour]
        merged=[item for pair in zip(x_norm, y_norm) for item in pair]
        points.extend(merged)
    return points
# def visualize_new_vs_ref(image_mean, alpha, step=30, B_value=255):
#     image_mean = np.array(image_mean, dtype=np.float32)
#     denom = 1 - alpha

#     # 建立所有可能的 ref_color 取樣
#     ref_rg = np.mgrid[0:256:step, 0:256:step].reshape(2, -1).T
#     ref_b = np.full((ref_rg.shape[0], 1), B_value)
#     ref_colors = np.hstack([ref_rg, ref_b]).astype(np.float32)

#     # 計算對應的 new_color
#     new_colors = alpha * image_mean + denom * ref_colors

#     # 篩選只保留合法範圍的 new_color
#     mask_valid = np.all((new_colors >= 0) & (new_colors <= 255), axis=1)
#     new_valid = new_colors[mask_valid]
#     ref_valid = ref_colors[mask_valid]

#     fig, ax = plt.subplots(figsize=(6, 6))
#     sc = ax.scatter(new_valid[:, 0], new_valid[:, 1],
#                     c=ref_valid[:, [2, 1, 0]] / 255.0,  # BGR → RGB
#                     s=40, marker='s', edgecolor='k')

#     ax.set_xlim(0, 255)
#     ax.set_ylim(0, 255)
#     ax.set_xlabel('new_color R')
#     ax.set_ylabel('new_color G')
#     ax.set_title(f'Legit new_color for B={B_value} (hover for info)')

#     # 使用 mplcursors 加入 hover 標籤顯示 new_color 與 ref_color
#     cursor = mplcursors.cursor(sc, hover=True)
#     @cursor.connect("add")
#     def on_add(sel):
#         idx = sel.index
#         new_c = new_valid[idx].astype(int)
#         ref_c = ref_valid[idx].astype(int)
#         sel.annotation.set_text(
#             f"new_color = ({new_c[0]}, {new_c[1]}, {int(new_c[2])})\n"
#             f"ref_color (BGR) = ({ref_c[0]}, {ref_c[1]}, {ref_c[2]})"
#         )
#         sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

#     plt.tight_layout()
#     plt.show()


def solve_setting_color(color, image_mean, alpha, eps=1e-6, mode="clip"):
    """
    根據希望呈現的 color，反推應該使用的 setting_color。
    如果無法實現（超出 [0,255]），根據 mode 處理：
      - "clip": 強行 clip 到 0~255（可能導致色偏）
      - "scale": 按比例縮放 RGB 向量（保色相方向）
    """
    color = np.array(color, dtype=np.float32)
    image_mean = np.array(image_mean, dtype=np.float32)

    # denom = 1 - alpha + eps
    # setting_color = (color - alpha * image_mean) / denom
    clip_bool=False
    if mode == "clip":
        setting_color = ((np.array(color) - alpha * image_mean)) / (1-alpha+eps)
        new_color = np.clip(setting_color, 0, 255)
        if any(element for element in setting_color if (element > 255 or element < 0)):
            clip_bool=True
        
    return new_color.astype(np.uint8), clip_bool
def yolo_transform(image, model, return_type='dict', plot_config=None, plot_key_list=None, show_plot_legend=True,  tolerance=0.5):
    # if return_type == 'image_array' and plot_config is None:
    #     raise ValueError("Provide a config for segmentation colors when return_type is 'image")
    if plot_config is None and return_type=='image_array':
        # with open('./conf/mask_color_setting.yaml', 'r') as file:
        #     plot_config = yaml.safe_load(file)
        raise ValueError("Provide a config for segmentation colors")
    # get the color list from config
    if plot_config is not None:
        # color_list=[plot_config['color_dict']]
        # color_list=[[color[2],color[1],color[0]] for color in color_list]
        #color_dict = {i: color for i, color in enumerate(color_list)}
        color_dict=plot_config['color_dict']
    
    plot_image=image.copy()
    results = model(image, verbose=False)
    class_names = model.names
    yolov8_contents=[]
    mask_dict={}
    error_message=''
    predict_label=None
    plot_alpha=0.3
    real_color_dict={}
    # 處理結果
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        if masks is None:
            error_message='Nothing detected for the image'
            continue
        predict_label=[]
        mask_colored = np.zeros_like(image)
        for i, (mask, box) in enumerate(zip(masks.data, boxes)):
            # Get class ID and confidence
            class_id = int(box.cls)
            confidence = float(box.conf)
            
            # Get class name
            class_name = class_names[class_id]
            if class_name not in predict_label:
                predict_label.append(class_name)
            # Convert mask to numpy array and resize to match original image
            mask_np = mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
            
            # Convert mask to binary image
            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
            if 'cvat' in return_type:
                contours = find_contours(mask_binary, 0.5)
                if len(contours)==0:
                    continue
                contour = contours[0]
                contour = np.flip(contour, axis=1)
                #polygons = approximate_polygon(contour, tolerance=tolerance)
                
                xyxy = box.xyxy.tolist()
                xtl = int(xyxy[0][0])
                ytl = int(xyxy[0][1])
                xbr = int(xyxy[0][2])
                ybr = int(xyxy[0][3])
                local_binary_mask=mask_binary[ytl:ybr, xtl:xbr]

                flatten_rle=mask_to_rle(local_binary_mask)
                flatten_rle_plus_bbox=flatten_rle + [xtl,ytl,xbr-1,ybr-1]

                if return_type=='cvat_mask':
                    cvat_mask = to_cvat_mask((xtl, ytl, xbr, ybr), mask_binary)
                    yolov8_contents.append({
                        "confidence": confidence,
                        "label": class_names[int(box.cls)],
                        "type": "mask",
                        "points": flatten_rle_plus_bbox,
                        "mask": cvat_mask
                    })
                else:
                    yolov8_contents.append({
                        "confidence": confidence,
                        "label": class_names[int(box.cls)],
                        "type": "mask",
                        "points": flatten_rle_plus_bbox,
                    })

            elif return_type=='yolov8':
                yolov8_points=get_yolov8_label(mask_binary, tolerance=tolerance)
                yolov8_line=[class_id]
                yolov8_line.extend(yolov8_points)
                if yolov8_points:
                    yolov8_contents.append(yolov8_line)
                # Check if the mask is valid
                if np.sum(mask_binary) == 0:
                    continue
                
                # Apply mask to the original image
                #mask_colored = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
            elif return_type=='dict':
                label=class_names[int(box.cls)]
                if mask_dict.get(label) is None:
                    mask_dict[label]=mask_binary
                else:
                    mask_dict[label]=cv2.bitwise_or(mask_dict[label], mask_binary) 
                

            if class_name != 'Background' and return_type == 'image_array':
                if plot_key_list is None or class_name in plot_key_list:
                    mask_colored[mask_binary == 255] = color_dict[class_name]
                    #no smmoothed so far
                    smoothed=mask_binary
                    #get the real color
                    blended =image[smoothed == 255]*plot_alpha+mask_colored[smoothed == 255]*(1-plot_alpha)
                    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
                    mean_RGB=blended_uint8.mean(axis=0).astype(np.uint8).tolist()
                    real_color_dict[class_name]=mean_RGB
                # Overlay the colored mask
                #plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)
                

    if return_type=="dict":
        return mask_dict

    if return_type=='image_array':
        present_labels=predict_label
        if not present_labels:
            return image, "No segmentation masks detected"
        if not show_plot_legend:
            return plot_image, error_message
        #plot image
        plot_image = cv2.addWeighted(image, plot_alpha, mask_colored, 1-plot_alpha, 0)
        # Load a custom or system font
        font = ImageFont.truetype("./conf/arial.ttf", 20)
        # find the plot text maximum width
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        max_text_width = 0
        for label in plot_config['color_dict']:
            if label in present_labels:
                text_width = dummy_draw.textlength(label, font=font)
                if text_width > max_text_width:
                    max_text_width = int(text_width)


        legend_width = int(image.shape[1]*0.078125)+max_text_width #100+max_text_width when width=1280
        block_height= int(image.shape[0]*0.03125) #30 when height=960
        legend_height = block_height * len(plot_config['color_dict']) 
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

        # Plot color blocks
        block_x1=int(legend_width*1/20)
        block_x2=int(legend_width*5/20)
        j = 0
        for label, color in plot_config['color_dict'].items():
            if label in present_labels:
                real_color=real_color_dict[label]
                cv2.rectangle(legend, (block_x1, j * block_height), (block_x2, (j + 1) * block_height), real_color, -1)
                j += 1

        # After drawing the color blocks, convert to a PIL image to draw the text (vector text)
        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)

        # Plot text
        text_width=int(legend_width*6/20)
        text_height=int(block_height)
        text_y_offset=int(block_height*1/6)
        j = 0
        for label, color in plot_config['color_dict'].items():
            if label in present_labels:
                draw.text((text_width, j * text_height + text_y_offset), label, font=font, fill=(255, 255, 255))
                j += 1

        # Convert back to a NumPy array for further processing
        legend = np.array(legend_pil)

        # Resize the legend and concatenate it with the plot image
        legend_resized = cv2.resize(legend, (int(legend_width * plot_image.shape[0] / legend.shape[0]), plot_image.shape[0]))
        concat_image = np.concatenate((plot_image, legend_resized), axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)
   
        return concat_image, error_message

    else:
        result_dict={
            'class_names': class_names,
            'yolov8_contents':yolov8_contents,
        }
        return result_dict


def pa_segmentation(image, model, model2, return_type, plot_config=None):

    result_dict={}
    model_1_select_key_list=['Alveolar_bone', 'Maxillary_sinus', 'Mandibular_alveolar_nerve', 'Periapical_lesion']
    model_2_select_key_list = [
        "Caries",
        "Crown",
        "Dentin",
        "Enamel",
        "Implant",
        "Post_and_core",
        "Pulp",
        "Restoration",
        "Root_canal_filling"
    ]
    
    if return_type=='yolov8':
        result1=yolo_transform(image, model, return_type, plot_config)
        result2=yolo_transform(image, model2, return_type, plot_config)
        key_index_mapping_1={value: key for key, value in result1['class_names'].items()}
        key_index_mapping_2={value: key for key, value in result2['class_names'].items()}
        #filter
        selected_indexes = [key_index_mapping_1[key] for key in model_1_select_key_list]
        result1_filtered = []
        for content in result1['yolov8_contents']:
            if content[0] in selected_indexes:
                label=result1['class_names'][content[0]]
                content[0]=key_index_mapping_2[label]
                result1_filtered.append(content)

        result2_filtered=[content for content in result1['yolov8_contents'] if content[0] in [key_index_mapping_2[index] for index in model_2_select_key_list]]
        result_dict['class_names']= result2['class_names']
        result_dict['yolov8_contents']=result1_filtered+result2_filtered
        return result_dict
            
    elif return_type=='image_array':
        error_message=''
        array_dict_1=yolo_transform(image, model, return_type='dict')
        array_dict_2=yolo_transform(image, model2, return_type='dict')
        if not array_dict_1 and not array_dict_2:
            return image, "Nothing detected for the image"
        
        mask_colored=np.zeros_like(image, dtype=np.uint8)
        present_labels=[]
        plot_alpha=0.5
        for label, mask in array_dict_1.items():
            if label in model_1_select_key_list:
                present_labels.append(label)
                filtered_mask=remove_small_regions(mask, min_area=int(0.0000813802*image.shape[0]*image.shape[1])) 
                smoothed = smooth_mask(filtered_mask, smoothing_factor= 10000, points_interp= 200)
                color=plot_config['color_dict'][label]
                image_mean=image[smoothed == 255].mean(axis=0)
                # print(label)
                # visualize_new_vs_ref(image_mean, plot_alpha, step=10)
                clipped_color, clip_bool = solve_setting_color(color, image_mean, plot_alpha, eps=1e-6, mode="clip")
                if clip_bool:
                    ref_color=((1-plot_alpha)*clipped_color+plot_alpha*image_mean).astype(np.uint8)
                    #print(f'label {label}, color {color} is not reachable with alpha {plot_alpha}, clipping with {clipped_color}, ref input color is {ref_color} ')
                mask_colored[smoothed == 255] = clipped_color

        for label, mask in array_dict_2.items():
            if label in model_2_select_key_list:
                present_labels.append(label)
                smoothed = smooth_mask(mask, smoothing_factor= 5000, points_interp= 200)
                color=plot_config['color_dict'][label]
                image_mean=image[smoothed == 255].mean(axis=0)
                
                clipped_color, clip_bool = solve_setting_color(color, image_mean, plot_alpha, eps=1e-6, mode="clip")
                if clip_bool:
                    ref_color=((1-plot_alpha)*clipped_color+plot_alpha*image_mean).astype(np.uint8)
                    #print(f'label {label}, color {color} is not reachable with alpha {plot_alpha}, clipping with {clipped_color}, ref input color is {ref_color} ')
                mask_colored[smoothed == 255] = clipped_color

        # Load a custom or system font
        font = ImageFont.truetype("./conf/arial.ttf", 20)
        # find the plot text maximum width
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        max_text_width = 0
        for label in plot_config['color_dict']:
            if label in present_labels:
                text_width = dummy_draw.textlength(label, font=font)
                if text_width > max_text_width:
                    max_text_width = int(text_width)

        plot_image = cv2.addWeighted(image, plot_alpha, mask_colored, 1-plot_alpha, 0)
        legend_width = int(image.shape[1]*0.078125)+max_text_width #100+max_text_width when width=1280
        block_height= int(image.shape[0]*0.03125) #30 when height=960
        legend_height = block_height * len(plot_config['color_dict']) 
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

        # Plot color blocks
        block_x1=int(legend_width*1/20)
        block_x2=int(legend_width*5/20)
        j = 0
        for label, color in plot_config['color_dict'].items():
            if label in present_labels:
                color=plot_config['color_dict'][label]
                #enhanced_color=adjust_hsv_weights_bgr()
                cv2.rectangle(legend, (block_x1, j * block_height), (block_x2, (j + 1) * block_height), color, -1)
                j += 1

        # After drawing the color blocks, convert to a PIL image to draw the text (vector text)
        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)

        # Plot text
        text_width=int(legend_width*6/20)
        text_height=int(block_height)
        text_y_offset=int(block_height*1/6)
        j = 0
        for label, color in plot_config['color_dict'].items():
            if label in present_labels:
                draw.text((text_width, j * text_height + text_y_offset), label, font=font, fill=(255, 255, 255))
                j += 1

        # Convert back to a NumPy array for further processing
        legend = np.array(legend_pil)

        # Resize the legend and concatenate it with the plot image
        legend_resized = cv2.resize(legend, (int(legend_width * plot_image.shape[0] / legend.shape[0]), plot_image.shape[0]))
        concat_image = np.concatenate((plot_image, legend_resized), axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)
        
        #show_two(concat_image, enhanced_image)
        return concat_image, error_message
    
    elif 'cvat' in return_type:
        result1=yolo_transform(image, model, return_type, plot_config)
        result2=yolo_transform(image, model2, return_type, plot_config)
        key_index_mapping_1={value: key for key, value in result1['class_names'].items()}
        key_index_mapping_2={value: key for key, value in result2['class_names'].items()}

        result1_filtered = [content for content in result1['yolov8_contents'] if content['label'] in model_1_select_key_list]
        result2_filtered = [content for content in result1['yolov8_contents'] if content['label'] in model_2_select_key_list]

        result_dict['class_names']= result2['class_names']
        result_dict['yolov8_contents']=result1_filtered+result2_filtered        
        return result_dict


if __name__=='__main__':
    model1=YOLO('./models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt')
    model2=YOLO('./models/dentistry_pa-segmentation_yolov11n-seg-all_25.20.pt')
    image=cv2.imread('./caries-1.166464-264-751_0_20220308129.png')
    with open('./conf/pa_segmentation_mask_color_setting.yaml', 'r') as file:
        config=yaml.safe_load(file)
    ###test code
    #test1, message=yolo_transform(image, model, return_type='yolov8', plot_config=config, tolerance=0.5)
    final_image, error_message=pa_segmentation(image, model1, model2, return_type='image_array' , plot_config=config)
    show_plot(final_image)
    #show_two(image, final_image)
    #final_image, _=yolo_transform(image, model1, return_type='image_array', plot_config=config, plot_key_list=None, show_plot_legend=True,  tolerance=0.5)
    #show_two(image,final_image)

    #show_plot(final_image)
    # test2=yolo_transform(image, return_type='cvat')
    # test3=yolo_transform(image, return_type='dict')

    #show_plot(test1)
