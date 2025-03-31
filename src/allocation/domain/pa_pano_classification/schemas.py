from pydantic import BaseModel
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator, Field

class PaPanoClassificationResponse(BaseModel):
    request_id: int  # 圖像的唯一標識符
    predicted_class: str  # 預測的類別
    scores: float  # 多個物體檢測結果
    message: str