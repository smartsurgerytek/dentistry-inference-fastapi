from pydantic import BaseModel, model_validator, ConfigDict
from typing import List, Tuple, Any
from fastapi import FastAPI, HTTPException
from typing import Annotated
from typing import Union, Tuple
from pydantic import BaseModel, Field
from typing_extensions import Literal
import ast
import base64
def read_base64_example():
    image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    
class DentalMeasurements(BaseModel):
    side_id: int = Field(..., ge=0, description="Side ID: 0 for left, 1 for right")
    CEJ: Tuple[int, int] = Field(..., description="cemento-enamel junction: the bottom of the crown position")
    ALC: Tuple[int, int] = Field(..., description="alveolar bone crest: the top of the alveolar bone")
    APEX: Tuple[int, int] = Field(..., description="cemento-enamel junction: the bottom of the teeth")
    CAL: float = Field(..., ge=0, description="cinical attachment Level (CAL). This is the length between the point between the CEJ and the ALC")
    TRL: float = Field(..., ge=0, description="Tooth Root Length. This is the distance from the CEJ to the APEX")
    ABLD: float = Field(..., ge=0, description="Alveolar Bone Loss Degree: ABLD (%) = ((Distance from CEJ to ALC − 2 mm) ÷ (Distance from CEJ to APEX − 2 mm)) × 100")
    stage: Union[Literal['0', "I", "II", "III"]]

class Measurements(BaseModel):
    teeth_id: int = Field(
        ..., 
        description="The identifier for the tooth from left to right, starting from 0."
    )
    pair_measurements: List["DentalMeasurements"] = Field(
        ..., 
        description="A list of line-based measurements from both the buccal and lingual sides of the tooth."
    )
    teeth_center: Tuple[int, int] = Field(
        ..., 
        description="The (x, y) pixel coordinates representing the center point of the tooth in the image."
    )
    
class PaMeasureDictResponse(BaseModel):
    request_id: int = Field(
        ..., 
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
        )
    )
    measurements: List["Measurements"] = Field(
        ..., 
        description=(
            "A list of structured measurement results, including `teeth_id`, `pair_measurements`, and `teeth_center`.\n\n"
            "**pair_measurements** refers to line-based measurements on both sides of a tooth, and includes:\n"
            "- `side_id`\n"
            "- `CEJ` (cemento-enamel junction): (x1, y1)\n"
            "- `ALC` (alveolar crest): (x1, y1)\n"
            "- `APEX` (tooth root apex): (x1, y1)\n"
            "- `CAL` (clinical attachment loss)\n"
            "- `TRL` (tooth root length)\n"
            "- `ABLD` (alveolar bone level distance)\n"
            "- `stage` (e.g., I, II, III)"
        )
    )
    message: str = Field(
        ..., 
        description="Status or informational message indicating the result of the measurement process. An empty string means no error occurred."
    )

class PaMeasureCvatResponse(BaseModel):
    request_id: int = Field(
        ..., 
        description="A unique identifier corresponding to the original measurement request. A value of 0 indicates that the result will not be stored in the database"
    )
    measurements: List[dict] = Field(
        ..., 
        description="List of measurement results. Each item represents the CVAT shapes format for points and polylines.\n\n"
    "### If label is `point`:\n"
    "- `label`, `type`, `points` [x1, y1], `teeth_id`, `side_id`\n\n"
    "### If label is `polyline`:\n"
    "- `label`, `type`, `points` [x1, y1, x2, y2], `teeth_id`, `side_id`\n"
    "- `attributes`: list of dicts with `name`, `input_type`, and `values`\n\n"
    "`attributes` with `name='stage'`: values can be `I`, `II`, `III`, or `'0'`"
    )
    message: str = Field(
        ..., 
        description="Informational or status message describing the result of the measurement process. No error will be returned empty string"
    )

#DentalMeasureDictResponse

class DentalMeasureDictValidator(BaseModel):
    image: bytes  # 图像字节
    scale_x: float  # 水平缩放比例
    scale_y: float  # 垂直缩放比例
    @model_validator(mode="before")
    def check_scale_range(cls, values):
        scale_x = values.get('scale_x')
        scale_y = values.get('scale_y')
        if scale_x is None or scale_y is None:
            raise ValueError(f"scale_x or scale_y must be not None")
        if not (0 <= scale_x <= 1):
            raise ValueError(f"scale_x must be in the range between 0 and 1: {scale_x}")
        if not (0 <= scale_y <= 1):
            raise ValueError(f"scale_y must be in the range between 0 and 1: {scale_y}")

        return values

class ImageResponse(BaseModel):
    request_id: int = Field(
        ..., 
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
        )
    )
    content_type: str = Field(
        ..., 
        description="The MIME type of the returned image (e.g., 'image/png')."
    )
    image: str = Field(
        ..., 
        description="Base64-encoded image string"
    )
    message: str = Field(
        ..., 
        description="Status or informational message. If successful, result in empty string."
    )


class PaMeasureRequest(BaseModel):
    image: str = Field(..., 
                       min_length=1, 
                       max_length=10_000_000, 
                       description="A Base64-encoded periapical (PA) radiographic image, commonly referred to as a bitewing film, used for periodontal assessment" \
                       " The image will be analyzed to detect anatomical landmarks, calculate distances and determine the periodontal stage")
    
    scale_x: float = Field(default=0.03125, ge=0.0, le=1.0, description="Scale factor for X-axis. Example:If the real-world wdith=40mm, pic width=1280 pixel then scale=40/1280=0.03125")
    scale_y: float = Field(default=0.03125, ge=0.0, le=1.0, description="Scale factor for Y-axis. Example:If the real-world height=30mm, pic height=960 pixel then scale=30/960=0.03125")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": read_base64_example(),
                "scale_x": 0.03125,
                "scale_y": 0.03125,
            }
        }
    )

class PaSegmentationRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=10_000_000)  # 增加最大長度限制
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": read_base64_example(),
            }
        }
    )