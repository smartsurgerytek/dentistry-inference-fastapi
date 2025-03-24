from pydantic import BaseModel, model_validator
from typing import List, Tuple, Any
from fastapi import FastAPI, HTTPException
from typing import Annotated
from typing import Union, Tuple
from pydantic import BaseModel, Field
from typing_extensions import Literal
import ast

    
class DentalMeasurements(BaseModel):
    side_id: int = Field(..., ge=0, description="Side ID must be a non-negative integer")
    CEJ: Tuple[int, int]
    ALC: Tuple[int, int]
    APEX: Tuple[int, int]
    CAL: float = Field(..., ge=0, description="CAL must be a non-negative float")
    TRL: float = Field(..., ge=0, description="TRL must be a non-negative float")
    ABLD: float = Field(..., ge=0, description="ABLD must be a non-negative float")
    stage: Union[Literal[0, 1, 2, 3, "I", "II", "III"]]

class Measurements(BaseModel):
    teeth_id: int
    pair_measurements: List[DentalMeasurements]
    teeth_center: Tuple[int, int]
    
class PaMeasureDictResponse(BaseModel):
    request_id: int
    measurements: List[Measurements]
    message: str

class PaMeasureCvatResponse(BaseModel):
    request_id: int
    measurements: List[dict]
    message: str

#DentalMeasureDictResponse

class DentalMeasureDictValidator(BaseModel):
    image: bytes  # 图像字节
    scale_x: float  # 水平缩放比例
    scale_y: float  # 垂直缩放比例
    @model_validator(mode="before")
    def check_scale_range(cls, values):
        scale_x = values.get('scale_x')
        scale_y = values.get('scale_y')
        if scale_x is None or scale_y is None:
            raise ValueError(f"scale_x or scale_y must be not None")
        if not (0 <= scale_x <= 1):
            raise ValueError(f"scale_x must be in the range between 0 and 1: {scale_x}")
        if not (0 <= scale_y <= 1):
            raise ValueError(f"scale_y must be in the range between 0 and 1: {scale_y}")

        return values

class ImageResponse(BaseModel):
    request_id: int
    content_type: str
    image: str# Base64 encoded string
    messages: str

class PaMeasureRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制
    scale_x: float = Field(default=1.0)
    scale_y: float = Field(default=1.0)

class PaSegmentationRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制