import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
from src.allocation.domain.dental_segmentation.main import *

def test_dentalSegmentation_normalImage():
    image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_dict=yolo_transform(image, return_type='dict')
    image=yolo_transform(image, return_type='image')
    if not results_dict.get('yolov8_contents'):
        raise ValueError("When input normal image, the result should be found, plz check dental_segmentation function")
    #show_plot(image)
    #print(results_dict)
def test_dentalSegmentation_blackimage():
    image=cv2.imread('./tests/files/black.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_dict=yolo_transform(image, return_type='dict')
    if results_dict.get('yolov8_contents'):
        raise ValueError("When input black image, the result should be [], plz check dental_segmentation function") 
    


if __name__ == "__main__":
    test_dentalSegmentation_normalImage()
    test_dentalSegmentation_blackimage()