from pydantic import BaseModel, ValidationError
from typing import List, Tuple, Any
from fastapi import FastAPI, HTTPException
from typing import Annotated
import ast
from pydantic.functional_validators import AfterValidator

def parse_tuple(value: str) -> Tuple[int, int]:
    try:
        # 使用 ast.literal_eval 來安全地解析字符串中的 tuple
        print(tuple(ast.literal_eval(value)))
        return tuple(ast.literal_eval(value))
    except Exception as e:
        raise ValueError(f"Invalid tuple format: {e}")
    
def check_len_2(value):
    if not len(value)==2:
        raise ValueError("len should be 2")
    
def check_tuple(value):
    if not isinstance(value, tuple):
        raise ValueError("scale after phaser remains invalid type")
    return value

def scale_phaser(value):
    if isinstance(value, str):
        value=parse_tuple(value)
    check_len_2(value)
    check_tuple(value)
    return value

scale_vlidator = Annotated[Any, AfterValidator(scale_phaser)]


class Scale(BaseModel):
    scale: scale_vlidator

print(Scale(scale='[1,2,3]'))