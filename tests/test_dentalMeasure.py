import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import cv2
from src.dental_measure.main import *
def test_dentalEstimation_normalImage():
    image=cv2.imread('./tests/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_list=dental_estimation(image, scale=(31/960,41/1080), return_type='dict')
    if not results_list:
        raise ValueError("When input normal image, the result should be found, plz check dental_estimation function")

def test_dentalEstimation_blackimage():
    image=cv2.imread('./tests/black.png')
    if image is None:
        raise ValueError("Image not found, check test image path (or utf8 problems) or cv2 package")
    results_list=dental_estimation(image, scale=(31/960,41/1080), return_type='dict')
    if results_list:
        raise ValueError("When input black image, the result should be [], plz check dental_estimation function")
