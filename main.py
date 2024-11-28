from fastapi import FastAPI, File, HTTPException, Form
from typing import Annotated
from src.schemas import InferenceResponse
from src.services.inference import InferenceService
import uvicorn
import ast
from typing import Tuple

def parse_tuple(value: str) -> Tuple[int, int]:
    try:
        # 使用 ast.literal_eval 來安全地解析字符串中的 tuple
        return ast.literal_eval(value)
    except Exception as e:
        raise ValueError(f"Invalid tuple format: {e}")
    
app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images."
)

@app.get("/", response_model=str)
async def read_root() -> str:
    return "Welcome to Smart Surgery Dentistry APIs!"

@app.post("/infer", response_model=InferenceResponse)
async def infer_dental_xray(
    image: Annotated[bytes, File()],
    scale, #: expected Annotated[str, Form()] or array
) -> InferenceResponse:
    scale=parse_tuple(scale)
    if not scale:
        raise HTTPException(status_code=400, detail="Scale is required")
    return InferenceService.process_xray(image, scale)
if __name__ == "__main__":
    uvicorn.run(app)