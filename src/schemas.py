from pydantic import BaseModel
from typing import List, Tuple

class Scale(BaseModel):
    scale: float

class Measurement(BaseModel):
    Id: int
    CEJ: Tuple[int, int]
    ALC: Tuple[int, int]
    APEX: Tuple[int, int]
    BL: float
    TR: float
    ABLD: float
    Stage: str

class InferenceResponse(BaseModel):
    RequestId: int
    Measurements: List[Measurement]
    Message: str
