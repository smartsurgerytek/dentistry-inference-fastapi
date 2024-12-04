from pydantic import BaseModel, validator
from typing import List, Tuple, Any, Dict
from fastapi import FastAPI, HTTPException
from typing import Annotated


from pydantic import BaseModel
from typing import List, Optional
import numpy as np

# # 定義 Bounding Box
# class BoundingBox(BaseModel):
#     x_min: float
#     y_min: float
#     x_max: float
#     y_max: float

# # 定義分割區域的多邊形（Segmentation Mask）
# # 假設我們將 segmentation 以多邊形的頂點座標表示，這些座標是以 [x, y] 格式的列表
# class SegmentationMask(BaseModel):
#     points: List[List[float]]  # List of [x, y] pairs

# # 定義物體檢測結果
class YoloSegmentation(BaseModel):
    color_dict: Dict[int,List[int]]
    class_names: Dict[int, str]
    yolov8_contents: List[Any]

# 定義 YOLO Segmentation 輸出結構
class YoloSegmentationResponse(BaseModel):
    request_id: int  # 圖像的唯一標識符
    yolo_results: YoloSegmentation  # 多個物體檢測結果
    message: str