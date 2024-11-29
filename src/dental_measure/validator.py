from pydantic import BaseModel, validator
from typing import List, Tuple, Any
import ast

class ScaleValidator(BaseModel):
    scale: Any
    @validator('scale')
    def validate_scale(cls, value):
        # 嘗試將字符串解析為元組
        try:
            parsed = tuple(ast.literal_eval(value))
        except Exception as e:
            raise ValueError(f"Invalid tuple format: {e}")
        
        # 確保元組的長度為 2
        if len(parsed) != 2:
            raise ValueError("Scale should have exactly 2 elements")
        
        # 確保值是元組類型
        if not isinstance(parsed, tuple):
            raise ValueError("Scale must be a tuple")
        
        return parsed