import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
from src.allocation.domain.dental_segmentation.main import *
import yaml
model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
def test_dentalSegmentation_normalImage():
    image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_dict=yolo_transform(image, model, return_type='dict')
    if not results_dict.get('yolov8_contents'):
        raise ValueError("When input normal image, the result should be found, plz check dental_segmentation function dict option")
    results_dict=yolo_transform(image, model, return_type='cvat')
    if not results_dict.get('yolov8_contents'):
        raise ValueError("When input normal image, the result should be found, plz check dental_segmentation function cvat option")

    with open('./conf/dentistry_PA.yaml', 'r') as file:
        config=yaml.safe_load(file)
    image=yolo_transform(image, model, return_type='image', config=config)
    

    #show_plot(image)
    #print(results_dict)
def test_dentalSegmentation_blackimage():
    image=cv2.imread('./tests/files/black.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_dict=yolo_transform(image, model, return_type='dict')
    if results_dict.get('yolov8_contents'):
        raise ValueError("When input black image, the result should be [], plz check dental_segmentation function") 
    


if __name__ == "__main__":
    test_dentalSegmentation_normalImage()
    test_dentalSegmentation_blackimage()