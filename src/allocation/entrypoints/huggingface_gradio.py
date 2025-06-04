import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import gradio as gr
import yaml
from PIL import Image
import cv2
from src.allocation.domain.pa_dental_measure.main import *
from src.allocation.domain.pa_dental_segmentation.main import *
from ultralytics import YOLO
from src.allocation.domain.pano_caries_detection.main import create_pano_caries_detection_model, pano_caries_detecion
from src.allocation.domain.pano_fdi_segmentation.main import pano_fdi_segmentation


title = "Dentistry Model segmentation Demo"
description = "Input A Image and get the segmentation result"
pa_segmentation_model = YOLO('./models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt')
pa_segmentation_model2 = YOLO('./models/dentistry_pa-segmentation_yolov11n-seg-all_25.20.pt')
pa_measurement_component_model=pa_segmentation_model
pa_measurement_contour_model=YOLO('./models/dentistry_pa-contour_yolov11x-seg_25.22.pt')
pano_caries_detection_model=create_pano_caries_detection_model(num_classes=1)
pano_caries_detection_wieght_path='./models/dentistry_pano-CariesDetection_resNetFpn_25.12.pth'

pano_fdi_segmentation_model=YOLO('./models/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt')
#test_function()
with open('./conf/pa_segmentation_mask_color_setting.yaml', 'r') as file:
    pa_segmentation_plot_config=yaml.safe_load(file)
def opencv2pil(opencv_image):
    return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
def create_inference(fn_name='pa_segmentation', label="Input Image", test_image_path=None):
    with gr.Row():
        image_input = gr.Image(label=label, value=cv2.imread(test_image_path) if test_image_path else None)
        image_output = gr.Image(label="Output Image")
    
    inference_button = gr.Button("Inference")

    if fn_name=='pa_segmentation':
        inference_button.click(
            lambda img: pa_segmentation(img, 
                                       model=pa_segmentation_model,
                                       model2=pa_segmentation_model2,
                                       return_type='image_array',
                                       plot_config=pa_segmentation_plot_config,
                                       )[0],
            inputs=image_input,
            outputs=image_output
        )
    elif fn_name=='pa_measurement':
        inference_button.click(
            lambda img: dental_estimation(img, 
                                          component_model=pa_measurement_component_model, 
                                          contour_model=pa_measurement_contour_model, 
                                          scale_x=31/960, 
                                          scale_y=41/1080, 
                                          return_type='image_array', 
                                          config=None)[0],
            inputs=image_input,
            outputs=image_output
        )
    elif fn_name=='pano_caries_detecion':
        inference_button.click(
            lambda img: pano_caries_detecion(pil_img=opencv2pil(img),
                                             model=pano_caries_detection_model, 
                                             weights_path=pano_caries_detection_wieght_path, 
                                             return_type='image_array')[0],
            inputs=image_input,
            outputs=image_output
        )
    elif fn_name=='pano_fdi_segmentation':
        inference_button.click(
            lambda img: pano_fdi_segmentation(img,
                                              model=pano_fdi_segmentation_model, 
                                              return_type='image_array')[0],
            inputs=image_input,
            outputs=image_output
        )

    return image_input, image_output, inference_button


demo = gr.Blocks()

with demo:
    gr.Markdown("Upload picture and get the image result")
    
    with gr.Tabs():
        with gr.TabItem("Periapical Film Segmentation"):
            create_inference(fn_name='pa_segmentation')
            create_inference(
                fn_name='pa_segmentation',
                label="Test Image 1",
                test_image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
            )
        with gr.TabItem("Periodontal measurements"):
            create_inference(fn_name='pa_measurement')
            create_inference(
                fn_name='pa_measurement',
                label="Test Image 1",
                test_image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
            )                        
        with gr.TabItem("PANO Caries Detection"):
            create_inference(fn_name='pano_caries_detecion')
            create_inference(
                fn_name='pano_caries_detecion',
                label="Test Image 1",
                test_image_path='./tests/files/027107.jpg'
            )

        with gr.TabItem("PANO FDI Segmentation"):
            create_inference(fn_name='pano_fdi_segmentation')
            create_inference(
                fn_name='pano_fdi_segmentation',
                label="Test Image 1",
                test_image_path='./tests/files/027107.jpg'
            )

demo.launch()