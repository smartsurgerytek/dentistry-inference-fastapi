import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
from src.allocation.domain.pa_dental_measure.main import *
component_model = YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
contour_model = YOLO('./models/dentistryContour_yolov11n-seg_4.46.pt')
def test_dentalEstimation_normalImage():
    image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_list=dental_estimation(image, component_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='dict', config=None)
    estimation_image=dental_estimation(image, component_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='image_array', config=None)
    if not results_list:
        raise ValueError("When input normal image, the result should be found, plz check dental_estimation function")
    #print(results_list)
    #show_plot(estimation_image)
def test_dentalEstimation_blackimage():
    image=cv2.imread('./tests/files/black.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_list=dental_estimation(image, component_model, contour_model, scale_x=31/960, scale_y=41 / 1080, return_type='dict', config=None)
    if results_list:
        raise ValueError("When input black image, the result should be [], plz check dental_estimation function")

if __name__ == '__main__':
    test_dentalEstimation_normalImage()
    test_dentalEstimation_blackimage()