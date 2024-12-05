from fastapi import FastAPI, Request, File, HTTPException, Form
from pydantic import ValidationError
from fastapi.responses import JSONResponse

from typing import Annotated
from src.dental_measure.schemas import InferenceResponse
from src.dental_segmentation.schemas import YoloSegmentationResponse
from src.services import InferenceService
import uvicorn
import ast
from typing import Tuple, Any
import ast
from pydantic.functional_validators import AfterValidator

from src.dental_measure.validator import ScaleValidator


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
            "message": "padantic model validation failed!",
            "details": exc.errors()
        }
    )

@app.get("/", response_model=str)
async def read_root() -> str:
    return "Welcome to Smart Surgery Dentistry APIs!"

@app.post("/periodontal_measure", response_model=InferenceResponse)
async def infer_dental_xray(
    image: Annotated[bytes, File()],
    scale: Any, #: expected Annotated[str, Form()] or array
) -> InferenceResponse:
    scale_obj=ScaleValidator(scale=scale)
    return InferenceService.process_xray(image, scale_obj.scale)

@app.post("/Dental_segmentation", response_model=YoloSegmentationResponse)
async def inference(
    image: Annotated[bytes, File()],
) -> YoloSegmentationResponse:
    return InferenceService.inference(image)

if __name__ == "__main__":
    uvicorn.run(app)