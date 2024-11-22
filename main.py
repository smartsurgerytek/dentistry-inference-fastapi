from fastapi import FastAPI, File, HTTPException, Form
from typing import Annotated

from src.schemas import InferenceResponse
from src.services.inference import InferenceService

app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images."
)

@app.post("/infer", response_model=InferenceResponse)
async def infer_dental_xray(
    image: Annotated[bytes, File()],
    scale: Annotated[float, Form()]
) -> InferenceResponse:
    if not scale:
        raise HTTPException(status_code=400, detail="Scale is required")

    return InferenceService.process_xray(image, scale)