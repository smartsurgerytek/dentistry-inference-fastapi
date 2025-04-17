from pydantic import BaseModel
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator, Field

# define yolo segmentation model
class YoloV8Segmentation(BaseModel):
    class_names: Dict[int, str] # class ID and class name
    yolov8_contents: List[List[Union[int, float]]] # List can contain both int and float
    @model_validator(mode="before")
    def validate_contents(cls, contents):
        for yolov8_label_list in contents['yolov8_contents']:
            if not isinstance(yolov8_label_list[0], int):
                raise ValueError("yolov8_label_list first element is class ID must be int")
        return contents    
# define yolo response
class PaSegmentationYoloV8Response(BaseModel):
    request_id: int  # 圖像的唯一標識符
    yolo_results: YoloV8Segmentation  # 多個物體檢測結果
    message: str


class CvatSegmentation(BaseModel):
    confidence: float
    label: str
    type: str
    points: List[int]
    #mask: List[int]

class CvatSegmentations(BaseModel):
    class_names: Dict[int, str] # class ID and class name
    yolov8_contents: List[CvatSegmentation]

class PaSegmentationCvatResponse(BaseModel):
    request_id: int  # 圖像的唯一標識符
    yolo_results: CvatSegmentations  # 多個物體檢測結果
    message: str

class PaSegmentationRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制
    class Config:
        json_schema_extra = {
            "example": {
                "image": ",/9j/4AAQSkZJRgABAQEAYA......",
            }
    }