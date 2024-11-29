from src.dental_measure.schemas import InferenceResponse, Measurements
from src.dental_measure.main import *
import numpy as np
import cv2
from io import BytesIO

class InferenceService:
    @staticmethod
    def process_xray_test(image: bytes) -> InferenceResponse:
        # Inference logic goes here
        return InferenceResponse(
            request_id=1,
            measurements=[
                Measurements(
                    Id=1,
                    CEJ=[100, 200],
                    ALC=[150, 250],
                    APEX=[200, 300],
                    BL=50.0,
                    TR=100.0,
                    ABLD=0.5,
                    Stage="Stage I"
                )
            ],
            message="Inference completed successfully"
        )
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

    # @staticmethod
    # def process_xray(image: bytes, scale: float) -> InferenceResponse:
    #     # Inference logic goes here
    #     return InferenceResponse(
    #         RequestId=1,
    #         Measurements=[
    #             Measurement(
    #                 Id=1,
    #                 CEJ=[100, 200],
    #                 ALC=[150, 250],
    #                 APEX=[200, 300],
    #                 BL=50.0,
    #                 TR=100.0,
    #                 ABLD=0.5,
    #                 Stage="Stage I"
    #             )
    #         ],
    #         Message="Inference completed successfully"
    #     )