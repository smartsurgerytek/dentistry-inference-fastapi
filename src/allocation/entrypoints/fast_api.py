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
from src.allocation.domain.dental_measure.schemas import PaMeasureDictResponse, ImageResponse, PaMeasureCvatResponse
from src.allocation.domain.dental_segmentation.schemas import PaSegmentationYoloV8Response, PaSegmentationCvatResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from src.allocation.adapters.utils import base64_to_bytes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global component_model
    component_model = YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
    
    global contour_model
    contour_model = YOLO('./models/dentistryContour_yolov11n-seg_4.46.pt')

    yield  
    # Cleanup on shutdown
    component_model = None
    contour_model = None

app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images.",
    lifespan=lifespan
)



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
    image: str,
    #scale: Any, #: expected Annotated[str, Form()] or array
    scale_x: float,
    scale_y: float,  
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(image)
    breakpoint()
    return InferenceService.pa_measure_dict(image, component_model, contour_model, scale_x, scale_y)

@app.post("/pa_measure_cvat", response_model=PaMeasureCvatResponse)
async def generate_periapical_film_measure_dict(
    image: str,
    #scale: Any, #: expected Annotated[str, Form()] or array
    scale_x: float,
    scale_y: float,  
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(image)
    return InferenceService.pa_measure_cvat(image, component_model, contour_model, scale_x, scale_y)
@app.post("/pa_measure_image", response_model=ImageResponse)#, response_model=DentalMeasureDictResponse)
async def generate_periapical_film_measure_image_base64(
    image: str,
    #scale: Any, #: expected Annotated[str, Form()] or array
    scale_x: float,
    scale_y: float,  
) -> ImageResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(image)
    return InferenceService.pa_measure_image_base64(image, component_model, contour_model, scale_x, scale_y)

@app.post("/pa_segmentation_yolov8", response_model=PaSegmentationYoloV8Response)
async def generate_periapical_film_segmentations_yolov8(
    image: str,
) -> PaSegmentationYoloV8Response:
    image=base64_to_bytes(image)
    return InferenceService.pa_segmentation_yolov8(image, component_model)

@app.post("/pa_segmentation_cvat", response_model=PaSegmentationCvatResponse)
async def generate_periapical_film_segmentations_cvat(
    image: str,
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(image)
    return InferenceService.pa_segmentation_cvat(image, component_model)

@app.post("/pa_segmentation_image", response_model=ImageResponse)
async def generate_periapical_film_segmentations_image_base64(
    image: str,
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(image)
    return InferenceService.pa_segmentation_image_base64(image, component_model)

if __name__ == "__main__":
    uvicorn.run(app)