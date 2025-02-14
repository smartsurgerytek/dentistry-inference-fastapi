from src.allocation.domain.dental_measure.schemas import PaMeasureDictResponse, DentalMeasureDictValidator, ImageResponse
from src.allocation.domain.dental_segmentation.schemas import *
from src.allocation.domain.dental_measure.main import *
from src.allocation.domain.dental_segmentation.main import *
import numpy as np
import cv2
from ultralytics import YOLO
class InferenceService:

    @staticmethod
    def pa_measure_dict(image: bytes, 
                        component_model:YOLO , 
                        contour_model:YOLO, 
                        scale_x: float, 
                        scale_y: float) -> PaMeasureDictResponse:
        
        validator=DentalMeasureDictValidator(image=image, scale_x=scale_x, scale_y=scale_y)

        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here

        measurements_list=dental_estimation(image=image_np, component_model=component_model, contour_model=contour_model, scale_x=scale_x, scale_y=scale_y, return_type='dict')

        if not measurements_list:
            return PaMeasureDictResponse(
                request_id=0,
                measurements=[],
                message="Nothing detected for the image"
            )
        
        return PaMeasureDictResponse(
            request_id=0,
            measurements=measurements_list,
            message="Inference completed successfully"
        )
    
    @staticmethod
    def pa_measure_image_base64(image: bytes, 
                                component_model:YOLO, 
                                contour_model:YOLO, 
                                scale_x: float, 
                                scale_y: float) -> ImageResponse:
        
        validator=DentalMeasureDictValidator(image=image, scale_x=scale_x, scale_y=scale_y)
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        output_image_array, error_message=dental_estimation(image=image_np, component_model=component_model, contour_model=contour_model, scale_x=scale_x, scale_y=scale_y, return_type='image_array')
        output_image_base64= numpy_to_base64(output_image_array, image_format='PNG')

        if error_message:
            return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            messages=error_message
        )

        return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            messages="Inference completed successfully"
        )
    
    @staticmethod
    def pa_segmentation_yolov8(image: bytes, model:YOLO) -> PaSegmentationYoloV8Response:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        yolov8_result_dict=yolo_transform(image=image_np, model= model, return_type='yolov8')
        if not yolov8_result_dict.get('yolov8_contents'):
            return PaSegmentationYoloV8Response(
                request_id=0,
                yolo_results=yolov8_result_dict,
                message="Nothing detected for the image"
            )
        return PaSegmentationYoloV8Response(
            request_id=0,
            yolo_results=yolov8_result_dict,
            message="Inference completed successfully"
        )