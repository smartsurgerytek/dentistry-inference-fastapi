import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fastapi import FastAPI, Request, File
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from typing import Annotated
from typing import Any
from typing import Optional
import uvicorn
from src.allocation.service_layer.services import InferenceService
from src.allocation.domain.pa_dental_measure.schemas import PaMeasureDictResponse, ImageResponse, PaMeasureCvatResponse, PaMeasureRequest
from src.allocation.domain.pa_dental_segmentation.schemas import PaSegmentationYoloV8Response, PaSegmentationCvatResponse, PaSegmentationRequest
from src.allocation.domain.pano_caries_detection.schemas import PanoCariesDetectionRequest, PanoCariesDetectionDictResponse
from src.allocation.domain.pano_caries_detection.main import create_pano_caries_detection_model
from src.allocation.domain.pa_pano_classification.main import create_pa_pano_classification_model
from src.allocation.domain.pa_pano_classification.schemas import PaPanoClassificationResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from src.allocation.adapters.utils import base64_to_bytes
from fastapi_swagger2 import FastAPISwagger2

import yaml
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global pa_component_model
    pa_component_model = YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
    
    global pa_contour_model
    pa_contour_model = YOLO('./models/dentistryContour_yolov11n-seg_4.46.pt')

    global pano_caries_detection_model
    pano_caries_detection_model= create_pano_caries_detection_model(1)

    global pano_caries_detection_model_weight_path
    pano_caries_detection_model_weight_path='./models/dentistry_pano-caries-detection-resNetFpn_5.12.pth'

    global pa_pano_classification_model
    pa_pano_classification_model=create_pa_pano_classification_model('./models/pa_pano_classification.pth')

    yield  
    # Cleanup on shutdown
    pa_component_model = None
    pa_contour_model = None
    pano_caries_detection_model= None
    pano_caries_detection_model_weight_path=''

app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images.",
    lifespan=lifespan
)
FastAPISwagger2(app)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "message": "pydantic model validation failed!",
            "details": [
                {
                    "loc": err["loc"],
                    "msg": err["msg"],
                    "type": err["type"]
                }
                for err in exc.errors()
            ]
        }
    )

@app.get("/", response_model=str)
async def read_root() -> str:
    return "Welcome to Smart Surgery Dentistry APIs!"

@app.post("/pa_measure_dict", response_model=PaMeasureDictResponse)
async def generate_periapical_film_measure_dict(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest,
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_dict(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)

@app.post("/pa_measure_cvat", response_model=PaMeasureCvatResponse)
async def generate_periapical_film_measure_dict(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest,
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_cvat(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)
@app.post("/pa_measure_image", response_model=ImageResponse)#, response_model=DentalMeasureDictResponse)
async def generate_periapical_film_measure_image_base64(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest
) -> ImageResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_image_base64(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)

@app.post("/pa_segmentation_yolov8", response_model=PaSegmentationYoloV8Response)
async def generate_periapical_film_segmentations_yolov8(
    #image: str,
    request: PaSegmentationRequest,
) -> PaSegmentationYoloV8Response:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_yolov8(image, pa_component_model)

@app.post("/pa_segmentation_cvat", response_model=PaSegmentationCvatResponse)
async def generate_periapical_film_segmentations_cvat(
    #image: str,
    request: PaSegmentationRequest
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_cvat(image, pa_component_model)

@app.post("/pa_segmentation_image", response_model=ImageResponse)
async def generate_periapical_film_segmentations_image_base64(
    request: PaSegmentationRequest
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_image_base64(image, pa_component_model)

@app.post("/pano_caries_detection_image", response_model=ImageResponse)
async def generate_pano_caries_detection_image_base64(
    request: PanoCariesDetectionRequest
) -> ImageResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_caries_detection_image_base64(image, pano_caries_detection_model, pano_caries_detection_model_weight_path)

@app.post("/pano_caries_detection_dict", response_model=PanoCariesDetectionDictResponse)
async def generate_pano_caries_detection_dict(
    request: PanoCariesDetectionRequest
) -> PanoCariesDetectionDictResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_caries_detection_dict(image, pano_caries_detection_model, pano_caries_detection_model_weight_path)

@app.post("/pa_pano_classification_dict", response_model=PaPanoClassificationResponse)
async def generate_pa_pano_classification(
    request: PaSegmentationRequest
) -> PaPanoClassificationResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_pano_classification_dict(image, pa_pano_classification_model)

if __name__ == "__main__":
    uvicorn.run(app)