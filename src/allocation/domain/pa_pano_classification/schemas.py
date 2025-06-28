from typing import List, Dict, Union
from pydantic import BaseModel, model_validator, Field

class PaPanoClassificationResponse(BaseModel):
    request_id: int = Field(
        ...,
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
        )
    )
    predicted_class: str = Field(
        ..., 
        description="The predicted classification label for the panoramic image (e.g., 'periapical film', 'panoramic x-ray', 'other')."
    )
    scores: float = Field(
        ..., 
        description="The confidence score (probability) associated with the predicted class, ranging from 0.0 to 1.0."
    )
    messages: str = Field(
        ..., 
        description="Status or informational message. If successful, result in empty string."
    )

