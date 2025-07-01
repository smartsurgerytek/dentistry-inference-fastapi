from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, model_validator, Field
import numpy as np
import base64
import yaml
def read_base64_example():
    with open("./conf/api_docs_setting.yaml", "r") as f:
        api_docs_setting = yaml.safe_load(f)
    # 取得 show_example_bool 的值
    show_example_bool = api_docs_setting.get("show_example_bool", False)
    if not show_example_bool:
        return "image: ,/9j/4AAQSkZJRgA......"
    image_path='./tests/files/027107.jpg'
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
class PanoCariesDetectionDict(BaseModel):
    boxes: List[List[int]] = Field(..., description="Bounding boxes in [x_min, y_min, x_max, y_max] format")
    labels: List[int] = Field(..., description="Class labels for each detected object")
    scores: List[float] = Field(..., description="Confidence scores for each detection")
    error_message: str = Field(
        ..., 
        description="Status or informational message. If successful, result in empty string."
    )
    
    # @classmethod
    # def from_torch(cls, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, error_message: Optional[str] = None):
    #     return cls(
    #         boxes=boxes.tolist(),
    #         labels=labels.tolist(),
    #         scores=scores.tolist(),
    #         error_message=error_message 
    #     )

class PanoCariesDetectionDictResponse(BaseModel):
    request_id: int = Field(
        ...,
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
        )
    )
    pano_caries_detection_dict: PanoCariesDetectionDict  
    message: str = Field(
        ..., 
        description=(
            "### Message Description\n"
            "This message describes the result of the postprocessing step in the process.\n\n"
            "**Possible values:**\n\n"
            "- `Inference completed successfully`: Inference completed normally.\n" \
            "- `No caries detected`: Inference completed normally but no caries found in the image. \n\n" \
        )
    )

class PanoCariesDetectionRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": read_base64_example(),
            }
        }
    )
    