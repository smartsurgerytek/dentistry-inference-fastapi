from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json



def plot_segmentations(image: np.ndarray, 
               class_name_list: list,
               mask_binary: np.ndarray, 
               class_id: int, 
               class_name: str, 
               plot_setting: dict
               ) -> np.ndarray:
    """
    Overlay class names and colored masks on the image based on segmentation results.
    
    Parameters:
    - image: np.ndarray, the original image array
    - class_name_list: list, list of class names that have been processed
    - mask_binary: np.ndarray, binary mask array for a single class
    - class_id: int, class ID
    - class_name: str, class name
    - plot_setting: dict, dictionary containing plotting settings such as font and color
    Returns:
    - np.ndarray, the image array with class names and masks overlaid
    - class_name_list: list, updated list of processed class names
    """    
    font_face = plot_setting['font_face']  # Font type
    font_scale = plot_setting['font_scale']  # Font size
    thickness = plot_setting['thickness']  # Text thickness
    line_type = plot_setting['line_type']  # Line type
    color_dict= plot_setting['color_dict']    

    plot_image=image.copy()
    # Apply mask to the original image
    mask_colored = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)

    if class_name != 'Background':# background label is no needing
        mask_colored[mask_binary == 255] = color_dict[str(class_id)]
        
        # Overlay the colored mask
        plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)
        
        # Calculate the average color of the masked area
        map_color = (plot_image[mask_binary == 255].sum(axis=0) / plot_image[mask_binary == 255].shape[0]).astype(np.uint8).tolist()
        
        # Add class name to the list if not already present
        if class_name not in class_name_list:
            class_name_list.append(class_name)

            # Draw class name on the image
            text_position = (mask_binary.shape[0] // 20, mask_binary.shape[1] // 10 * (len(class_name_list)))
            cv2.putText(plot_image, class_name, text_position, font_face, font_scale, map_color, thickness, line_type)
    
    return plot_image,class_name_list

def retrieve_yolo_results(result: 'ultralytics.YOLOResult', image: np.ndarray, plot_setting: dict, return_type='plot') -> None:
    """
    Plots the YOLO detection results on an input image with customizable settings.

    Args:
        result (dict): The detection results from the YOLO model, typically containing bounding boxes, class labels, and confidence scores.
        image (np.ndarray): The input image in OpenCV format (a NumPy array) on which the results will be plotted.
        plot_setting (dict): A dictionary containing plot settings such as colors, line thickness, and display options.

    Returns:
        None: The function displays the image with detected objects overlaid according to the specified plot settings. No value is returned.

    Notes:
        Ensure that the `result` dictionary is correctly formatted and matches the output of the YOLO model. Verify that `plot_setting` includes valid configuration parameters for visualization.
    """

    # 處理結果
    detections=[]
    # 獲取類別名稱
    class_names = result.names
    class_name_list=[]         
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    plot_image=image.copy()
    if masks is None:
        return None
    for mask, box in zip(masks.data, boxes):
        # Get class ID and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)
        
        # Get class name
        class_name = class_names[class_id]

        # Convert mask to numpy array and resize to match original image
        mask_np = mask.cpu().numpy()
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        
        # Convert mask to binary image
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
        
        # Check if the mask is valid
        if np.sum(mask_binary) == 0:
            continue

        plot_image, class_name_list=plot_segmentations(plot_image, 
                                                       class_name_list,
                                                       mask_binary,
                                                       class_id, 
                                                       class_name, 
                                                       plot_setting,
                                                    )
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "segmentation_mask": mask_binary.tolist()
        })

    if return_type=='plot':
        return plot_image
    elif return_type=='dict':
        return detections
    else:
        assert False, 'check the return type either plot or dict'


if __name__ == '__main__':
    image_path='./data/pics/caries-0.6741573-260-760_1_2022052768.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = YOLO('./dentistry_yolov8n_20240807_all.pt')

    results=model(image)

    with open('./conf/mask_color_setting.json', 'r') as file:
        plot_setting = json.load(file)

    result=results[0]
    image_plot=retrieve_yolo_results(result, image, plot_setting, 'plot')

    plt.imshow(image_plot)
    plt.show()