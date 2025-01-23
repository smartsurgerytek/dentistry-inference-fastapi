from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.measure import approximate_polygon
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from src.allocation.domain.dental_segmentation.utils import *
import yaml

model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
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

def yolo_transform(image, return_type='dict', config=None):
    if return_type == 'image' and config is None:
        raise ValueError("Provide a config for segmentation colors when return_type is 'image")

    # 定義文本參數
    color_list=config['color_list']
    #color_list=[[color[2],color[1],color[0]] for color in color_list]
    plot_image=image.copy()
    color_dict = {i: color for i, color in enumerate(color_list)}

    results = model(image)

    # 獲取類別名稱
    class_names = model.names
    yolov8_contents=[]
    # 處理結果
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        if masks is None:
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
            yolov8_points=get_yolov8_label(mask_binary, tolerance=0.5)
            yolov8_line=[class_id]
            yolov8_line.extend(yolov8_points)
            if yolov8_points:
                yolov8_contents.append(yolov8_line)
            #breakpoint()
            # Check if the mask is valid
            if np.sum(mask_binary) == 0:
                continue
            
            # Apply mask to the original image
            mask_colored = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
            if class_name != 'Background' and return_type=='image':
                mask_colored[mask_binary == 255] = color_dict[class_id]
                # Overlay the colored mask
                plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)
                
                
    if return_type=='image':
        label_image=get_label_text_img(result.boxes.cls.cpu().numpy().astype(int), plot_image.shape[1], color_dict, class_names)

        plot_image=np.concatenate((plot_image, label_image), axis=0)

        plot_image = cv2.resize(plot_image, (image.shape[1], image.shape[0]))
        return plot_image
    
    else:
        result_dict={
            'color_dict': color_dict,
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
    image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    with open('./conf/dentistry_PA.yaml', 'r') as file:
        config=yaml.safe_load(file)
    tests=yolo_transform(image, return_type='image', config=config)
    #show_plot(result)
