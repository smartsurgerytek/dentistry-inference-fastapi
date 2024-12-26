import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import gradio as gr
import cv2
from src.allocation.domain.dental_measure.main import *
from src.allocation.domain.dental_segmentation.main import *
title = "Dentistry Model segmentation Demo"
description = "Input A Image and get the segmentation result"


#test_function()
def create_inference(fn_name='inference', label="Input Image", test_image_path=None):
    with gr.Row():
        image_input = gr.Image(label=label, value=cv2.imread(test_image_path) if test_image_path else None)
        image_output = gr.Image(label="Output Image")
    
    inference_button = gr.Button("Inference")

    if fn_name=='inference':
        inference_button.click(
            lambda img: yolo_transform(image=img, return_type='image'),
            inputs=image_input,
            outputs=image_output
        )
    elif fn_name=='measurement':
        inference_button.click(
            dental_estimation,
            inputs=image_input,
            outputs=image_output
        )    
    return image_input, image_output, inference_button


demo = gr.Blocks()

with demo:
    gr.Markdown("Upload periapical film picture")
    
    with gr.Tabs():
        with gr.TabItem("Model inference"):
            create_inference(fn_name='inference')
            create_inference(
                fn_name='inference',
                label="Test Image 1",
                test_image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
            )
        with gr.TabItem("Periodontal measurements"):
            create_inference(fn_name='measurement')
            create_inference(
                fn_name='measurement',
                label="Test Image 1",
                test_image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
            )                        


demo.launch()