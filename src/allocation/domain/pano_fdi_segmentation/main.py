from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from src.allocation.domain.pa_dental_segmentation.main import yolo_transform
from src.allocation.domain.pa_dental_measure.utils import *
import yaml



def pano_fdi_segmentation(image, model, plot_config=None, return_type='cvat'):
    if plot_config is None:
        with open('./conf/pano_fdi_segmentation_mask_color_setting.yaml', 'r') as file:
            plot_config = yaml.safe_load(file)

    if return_type=='image_array':
        plot_image, error_message=yolo_transform(image, model, return_type=return_type, plot_config=plot_config, tolerance=0.5)
        return plot_image, error_message
    else:
        result_dict=yolo_transform(image, model, return_type=return_type, plot_config=plot_config, tolerance=0.5)
        return result_dict
    
if __name__=='__main__':
    model=YOLO('./models/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt')
    image=cv2.imread('./tests/files/027107.jpg')
    image, messages =pano_fdi_segmentation(image, model, return_type='image_array')
    show_plot(image)
    result_dict=pano_fdi_segmentation(image, model, plot_config=None, return_type='cvat')
    print(result_dict)
