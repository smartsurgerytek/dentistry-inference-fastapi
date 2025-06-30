from src.allocation.domain.pa_dental_measure.schemas import PaMeasureDictResponse, DentalMeasureDictValidator, ImageResponse, PaMeasureCvatResponse
from src.allocation.domain.pa_dental_segmentation.schemas import *
from src.allocation.domain.pa_dental_measure.main import *
from src.allocation.domain.pa_dental_segmentation.main import *
from src.allocation.domain.pano_caries_detection.main import *
from src.allocation.domain.pano_caries_detection.schemas import *
from src.allocation.domain.pa_pano_classification.main import *
from src.allocation.domain.pa_pano_classification.schemas import *
from src.allocation.domain.pano_fdi_segmentation.main import *
from src.allocation.domain.pano_fdi_segmentation.schemas import *
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import yaml
import io
def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
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
    def pa_measure_cvat(image: bytes, 
                        component_model:YOLO , 
                        contour_model:YOLO, 
                        scale_x: float, 
                        scale_y: float):
        
        validator=DentalMeasureDictValidator(image=image, scale_x=scale_x, scale_y=scale_y)
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here

        measurements_list=dental_estimation(image=image_np, component_model=component_model, contour_model=contour_model, scale_x=scale_x, scale_y=scale_y, return_type='cvat')

        if not measurements_list:
            return PaMeasureCvatResponse(
                request_id=0,
                measurements=[],
                message="Nothing detected for the image"
            )
        
        return PaMeasureCvatResponse(
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
            message=error_message
        )

        return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message="Inference completed successfully"
        )
    
    @staticmethod
    def pa_segmentation_yolov8(image: bytes, model:YOLO, model2:YOLO) -> PaSegmentationYoloV8Response:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        #yolov8_result_dict=yolo_transform(image=image_np, model= model, return_type='yolov8')
        yolov8_result_dict=pa_segmentation(image=image_np, model= model, model2= model2, return_type='yolov8', plot_config=None)

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
    
    @staticmethod
    def pa_segmentation_cvat(image: bytes, model:YOLO, model2:YOLO) -> PaSegmentationCvatResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        #cvat_result_dict=yolo_transform(image=image_np, model= model, return_type='cvat')
        cvat_result_dict=pa_segmentation(image=image_np, model= model, model2= model2, return_type='cvat', plot_config=None)
        #drop the mask cols in cvat_result_dict
        # for sublist in cvat_result_dict['yolov8_contents']:
        #     if 'mask' in sublist:
        #         del sublist['mask']  # 或者用 pop() 方法: sublist.pop('mask', None)

        if not cvat_result_dict.get('yolov8_contents'):
            return PaSegmentationCvatResponse(
                request_id=0,
                yolo_results=cvat_result_dict,
                message="Nothing detected for the image"
            )
        return PaSegmentationCvatResponse(
            request_id=0,
            yolo_results=cvat_result_dict,
            message="Inference completed successfully"
        )    

    @staticmethod
    def pa_segmentation_image_base64(image: bytes, model:YOLO, model2:YOLO) -> ImageResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        plot_config=read_yaml('./conf/pa_segmentation_mask_color_setting.yaml')
        #output_image_array, error_message=yolo_transform(image=image_np, model= model, return_type='image_array', plot_config=plot_config)
        output_image_array, error_message=pa_segmentation(image=image_np, model= model, model2= model2, return_type='image_array', plot_config=plot_config)
        #drop the mask cols in cvat_result_dict
        #show_plot(output_image_array)

        output_image_base64= numpy_to_base64(output_image_array, image_format='PNG')

        if error_message:
            return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message=error_message
        )

        return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message="Inference completed successfully"
        )

    @staticmethod
    def pano_caries_detection_image_base64(image: bytes, model, weights_path: str ) -> ImageResponse:
        image_pil= Image.open(io.BytesIO(image))

        output_image_array, error_message = pano_caries_detecion(model, weights_path, image_pil, return_type='image_array')
        #drop the mask cols in cvat_result_dict
        #show_plot(output_image_array)


        output_image_base64= numpy_to_base64(output_image_array, image_format='PNG')
        
        if error_message:
            return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message=error_message
        )        


        return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message="Inference completed successfully"
        )
    
    @staticmethod
    def pano_caries_detection_dict(image: bytes, model, weights_path: str ) -> PanoCariesDetectionDictResponse:
        image_pil= Image.open(io.BytesIO(image))
        results_dict = pano_caries_detecion(model, weights_path, image_pil, return_type='dict')

        if not results_dict['error_message']:
            return PanoCariesDetectionDictResponse(
                request_id=0,
                pano_caries_detection_dict=results_dict,  
                message="Inference completed successfully"
            )
        else:
            return PanoCariesDetectionDictResponse(
                request_id=0,
                pano_caries_detection_dict=results_dict,  
                message="No caries detected"
            )
        
    @staticmethod
    def pa_pano_classification_dict(image: bytes, model) -> ImageResponse:
        image_pil= Image.open(io.BytesIO(image)).convert('RGB')
        predicted_class, scores = predict_image_pa_pano_classification(model, image_pil)  
        return PaPanoClassificationResponse(
            request_id=0,
            predicted_class=predicted_class,
            scores=scores,
            message="Classification completed successfully"
        )
    
    @staticmethod
    def pano_fdi_segmentation_image_base64(image: bytes, model) -> ImageResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
        output_image_array, error_message=pano_fdi_segmentation(image_np, model, return_type='image_array')
        output_image_base64= numpy_to_base64(output_image_array, image_format='PNG')

        if error_message:
            return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message=error_message
        )

        return ImageResponse(
            request_id=0,
            image=output_image_base64,
            content_type='image/png',
            message="Inference completed successfully"
        )        
    @staticmethod
    def pano_fdi_segmentation_cvat(image: bytes, model:YOLO) -> PanoSegmentationCvatResponse:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        cvat_result_dict=pano_fdi_segmentation(image_np, model, return_type='cvat')

        if not cvat_result_dict.get('yolov8_contents'):
            return PanoSegmentationCvatResponse(
                request_id=0,
                yolo_results=cvat_result_dict,
                message="Nothing detected for the image"
            )
        return PanoSegmentationCvatResponse(
            request_id=0,
            yolo_results=cvat_result_dict,
            message="Inference completed successfully"
        )
    
    @staticmethod
    def pano_fdi_segmentation_yolov8(image: bytes, model:YOLO) -> PanoSegmentationYoloV8Response:
        image_np = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)# Inference logic goes here
        yolov8_result_dict=pano_fdi_segmentation(image_np, model, return_type='yolov8')

        if not yolov8_result_dict.get('yolov8_contents'):
            return PanoSegmentationYoloV8Response(
                request_id=0,
                yolo_results=yolov8_result_dict,
                message="Nothing detected for the image"
            )
        return PanoSegmentationYoloV8Response(
            request_id=0,
            yolo_results=yolov8_result_dict,
            message="Inference completed successfully"
        )