from pydantic import BaseModel
from typing import List, Dict, Union
from pydantic import BaseModel


# define yolo segmentation model
class YoloSegmentation(BaseModel):
    color_dict: Dict[int,List[int]] # class ID and color RGB
    class_names: Dict[int, str] # class ID and class name
    yolov8_contents: List[Union[int, float]] # List can contain both int and float

# define yolo response
class YoloSegmentationResponse(BaseModel):
    request_id: int  # 圖像的唯一標識符
    yolo_results: YoloSegmentation  # 多個物體檢測結果
    message: str