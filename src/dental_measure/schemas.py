from pydantic import BaseModel, validator
from typing import List, Tuple, Any
from fastapi import FastAPI, HTTPException
from typing import Annotated
import ast

    
class DentalMeasurements(BaseModel):
    side_id: int
    CEJ: Tuple[int, int]
    ALC: Tuple[int, int]
    APEX: Tuple[int, int]
    CAL: float
    TRL: float
    ABLD: float
    stage: str

class Measurements(BaseModel):
    teeth_id: int
    pair_measurements: List[DentalMeasurements]
    teeth_center: Tuple[int, int]
    
class InferenceResponse(BaseModel):
    request_id: int
    measurements: List[Measurements]
    message: str

