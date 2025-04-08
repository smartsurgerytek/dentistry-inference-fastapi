from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, model_validator, Field
import numpy as np

class PanoCariesDetectionDict(BaseModel):
    boxes: List[List[int]] = Field(..., description="Bounding boxes in [x_min, y_min, x_max, y_max] format")
    labels: List[int] = Field(..., description="Class labels for each detected object")
    scores: List[float] = Field(..., description="Confidence scores for each detection")
    error_messages: Optional[str] = Field(default=None, description="List of error messages, if any")
    
    # @classmethod
    # def from_torch(cls, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, error_messages: Optional[str] = None):
    #     return cls(
    #         boxes=boxes.tolist(),
    #         labels=labels.tolist(),
    #         scores=scores.tolist(),
    #         error_messages=error_messages 
    #     )

class PanoCariesDetectionDictResponse(BaseModel):
    request_id: int  # 圖像的唯一標識符
    pano_caries_detection_dict: PanoCariesDetectionDict  
    message: str

class PanoCariesDetectionRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制
    class Config:
        json_schema_extra = {
            "example": {
                "image": ",/9j/4AAQSkZJRgABAQEAYA......",
            }
    }
    