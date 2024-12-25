from src.allocation.domain.dental_measure.schemas import InferenceResponse, Measurements
from src.allocation.domain.dental_segmentation.schemas import *
from src.allocation.domain.dental_measure.main import *
from src.allocation.domain.dental_segmentation.main import *
import numpy as np
import cv2

class InferenceService:

    @staticmethod
    def process_xray(image: bytes, scale: tuple) -> InferenceResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        measurements_list=dental_estimation(image_np, scale=scale, return_type='dict')
        #print(measurements_list)
        if not measurements_list:
            return InferenceResponse(
                request_id=0,
                measurements=[],
                message="Nothing detected for the image"
            )
        
        return InferenceResponse(
            request_id=0,
            measurements=measurements_list,
            message="Inference completed successfully"
        )
    @staticmethod
    def inference(image: bytes) -> YoloSegmentationResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        yolov8_result_dict=yolo_transform(image_np, return_type='dict')

        if not yolov8_result_dict.get('yolov8_contents'):
            return YoloSegmentationResponse(
                request_id=0,
                yolo_results=yolov8_result_dict,
                message="Nothing detected for the image"
            )
        
        return YoloSegmentationResponse(
            request_id=0,
            yolo_results=yolov8_result_dict,
            message="Inference completed successfully"
        )