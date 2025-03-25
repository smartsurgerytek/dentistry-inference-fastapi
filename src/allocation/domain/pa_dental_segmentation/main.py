from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.measure import approximate_polygon
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from src.allocation.domain.pa_dental_segmentation.utils import *
import yaml

#model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
def find_center_mask(mask_binary):
    moments = cv2.moments(mask_binary)

    # 計算質心
    if moments['m00'] != 0:  # 確保掩膜不全為零
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = None, None  # 如果掩膜全為零
    return (cx,cy)


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

def yolo_transform(image, model, return_type='dict', plot_config=None, tolerance=0.5):
    # if return_type == 'image_array' and plot_config is None:
    #     raise ValueError("Provide a config for segmentation colors when return_type is 'image")
    if plot_config is None:
        with open('./conf/mask_color_setting.yaml', 'r') as file:
            plot_config = yaml.safe_load(file)
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
    # 處理結果
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        if masks is None:
            error_message='No segmenation masks detected'
            continue
        predict_label=[]
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
            mask_colored = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
            if return_type=='cvat':
                contours = find_contours(mask_binary, 0.5)
                if len(contours)==0:
                    continue
                contour = contours[0]
                contour = np.flip(contour, axis=1)
                polygons = approximate_polygon(contour, tolerance=tolerance)

                xyxy = box.xyxy.tolist()
                xtl = int(xyxy[0][0])
                ytl = int(xyxy[0][1])
                xbr = int(xyxy[0][2])
                ybr = int(xyxy[0][3])

                #cvat_mask = to_cvat_mask((xtl, ytl, xbr, ybr), mask_binary)

                yolov8_contents.append({
                    "confidence": confidence,
                    "label": class_names[int(box.cls)],
                    "type": "mask",
                    "points": polygons.ravel().tolist(),
                    #"mask": cvat_mask
                })
            elif return_type=="cvat_mask": 
                contours = find_contours(mask_binary, 0.5)
                if len(contours)==0:
                    continue
                contour = contours[0]
                contour = np.flip(contour, axis=1)
                polygons = approximate_polygon(contour, tolerance=tolerance)

                xyxy = box.xyxy.tolist()
                xtl = int(xyxy[0][0])
                ytl = int(xyxy[0][1])
                xbr = int(xyxy[0][2])
                ybr = int(xyxy[0][3])

                cvat_mask = to_cvat_mask((xtl, ytl, xbr, ybr), mask_binary)

                yolov8_contents.append({
                    "confidence": confidence,
                    "label": class_names[int(box.cls)],
                    "type": "mask",
                    "points": polygons.ravel().tolist(),
                    "mask": cvat_mask
                })                               
            elif return_type=='yolov8':
                yolov8_points=get_yolov8_label(mask_binary, tolerance=tolerance)
                yolov8_line=[class_id]
                yolov8_line.extend(yolov8_points)
                if yolov8_points:
                    yolov8_contents.append(yolov8_line)
                #breakpoint()
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
                

            if class_name != 'Background' and return_type=='image_array':
                mask_colored[mask_binary == 255] = color_dict[class_name]
                # Overlay the colored mask
                plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)

    if return_type=="dict":
        return mask_dict
                
                
    if return_type=='image_array':
        label_image=get_label_text_img(result.boxes.cls.cpu().numpy().astype(int), plot_image.shape[1], color_dict, class_names)

        plot_image=np.concatenate((plot_image, label_image), axis=0)

        plot_image = cv2.resize(plot_image, (image.shape[1], image.shape[0]))
        return plot_image, error_message
    
    else:
        result_dict={
            'class_names': class_names,
            'yolov8_contents':yolov8_contents,
        }
        return result_dict

def show_plot(image):
    #cv2.imshow("OpenCV Image", image)
    # 使用 matplotlib 绘制图形
    plt.figure()
    plt.imshow(image)
    plt.show()


if __name__=='__main__':
    #model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
    image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    with open('./conf/dentistry_PA.yaml', 'r') as file:
        config=yaml.safe_load(file)
    test1=yolo_transform(image, return_type='image', plot_config=config)
    test2=yolo_transform(image, return_type='cvat')
    test3=yolo_transform(image, return_type='dict')

    #show_plot(result)
