from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.measure import approximate_polygon
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dental_segmentation.utils import *


model=YOLO('./models/dentistry_yolov11x-seg_4.42.pt')
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

def yolo_transform(image, return_type='dict'):
    # 定義文本參數
    image= cv2.resize(image, (1280,960))
    plot_image=image.copy()
    color_list = [
        [0, 240, 255],
        [65, 127, 0],
        [0, 0, 255],
        [113, 41, 29],
        [122, 21, 135],
        [0, 148, 242],
        [4, 84, 234],
        [0, 208, 178],
        [52, 97, 148],
        [121, 121, 121],
        [212, 149, 27],
        [206, 171, 255],
        [110, 28, 216]
    ]

    color_list=[[color[2],color[1],color[0]] for color in color_list]
    color_dict = {i: color for i, color in enumerate(color_list)}
    text_dict={
    0: "Alveolar_bone",
    1: "Caries",
    2: "Crown",
    3: "Dentin",
    4: "Enamel",
    5: "Implant",
    6: "Mandibular_alveolar_nerve",
    7: "Maxillary_sinus",
    8: "Periapical_lesion",
    9: "Post_and_core",
    10: "Pulp",
    11: "Restoration",
    12: "Root_canal_filling"
    }

    # image_path = './M_02172016084007_0000000S1555078C_2_001_001-003-1 cutX2_001539000.jpg'
    # image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
            if class_name != 'Background':
                mask_colored[mask_binary == 255] = color_dict[class_id]
                # Overlay the colored mask
                plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)
                
                
    if return_type=='image':
        label_image=get_label_text_img(result.boxes.cls.cpu().numpy().astype(int), plot_image.shape[1], color_dict, text_dict)

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
    image=cv2.imread('./tests/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    tests=yolo_transform(image, return_type='dict')
    #show_plot(result)
