from pydantic import BaseModel
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator

# define yolo segmentation model
class YoloV8SegmentationResponse(BaseModel):
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
    yolo_results: YoloV8SegmentationResponse  # 多個物體檢測結果
    message: str