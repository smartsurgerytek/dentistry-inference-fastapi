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
from src.allocation.domain.dental_measure.schemas import PaMeasureDictResponse, ImageResponse
from src.allocation.domain.dental_segmentation.schemas import PaSegmentationYoloV8Response


app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images."
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
    image: Annotated[bytes, File()],
    #scale: Any, #: expected Annotated[str, Form()] or array
    scale_x: float,
    scale_y: float,  
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    return InferenceService.pa_measure_dict(image, scale_x, scale_y)

@app.post("/pa_measure_image", response_model=ImageResponse)#, response_model=DentalMeasureDictResponse)
async def generate_periapical_film_measure_image_base64(
    image: Annotated[bytes, File()],
    #scale: Any, #: expected Annotated[str, Form()] or array
    scale_x: float,
    scale_y: float,  
) -> ImageResponse:
    #scale_obj=ScaleValidator(scale=scale)
    return InferenceService.pa_measure_image_base64(image, scale_x, scale_y)

@app.post("/pa_segmentation_yolov8", response_model=PaSegmentationYoloV8Response)
async def generate_periapical_film_segmentations_yolov8(
    image: Annotated[bytes, File()],
) -> PaSegmentationYoloV8Response:
    return InferenceService.pa_segmentation_yolov8(image)

if __name__ == "__main__":
    uvicorn.run(app)