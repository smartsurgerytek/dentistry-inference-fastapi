from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator, Field
import base64
def read_base64_example():
    image_path='./tests/files/caries-0.6741573-260-760_1_2022052768.png'
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
# define yolo segmentation model
class YoloV8Segmentation(BaseModel): #note that any desription in this will cause generating api docs bugs
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
    request_id: int = Field(
        ...,
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
            "Detection results from YOLOv8 segmentation model:\n\n"
            "**class_names:** Mapping from class ID to class name and anatomical description:\n\n"
            "- 0: Crown — The visible part of the tooth above the gumline, covered by enamel, responsible for biting and chewing.\n"
            "- 1: Alveolar_bone — The bone structure surrounding and supporting the teeth within the jaw.\n"
            "- 2: Caries — Areas of tooth decay caused by bacterial activity leading to demineralization.\n"
            "- 3: Dentin — The dense, bony tissue beneath the enamel forming the bulk of the tooth.\n"
            "- 4: Pulp — The innermost part of the tooth containing nerves and blood vessels.\n"
            "- 5: Maxillary_sinus — Air-filled spaces located within the bones of the upper jaw near the nose.\n"
            "- 6: Implant — A dental prosthesis surgically inserted into the jawbone to replace missing teeth.\n"
            "- 7: Enamel — The hard, outermost layer of the tooth, protecting it from decay.\n"
            "- 8: Post_and_core — Dental restoration components used to rebuild a tooth's structure after root canal treatment.\n"
            "- 9: Restoration — Materials or procedures used to repair damaged teeth (e.g., fillings, crowns).\n"
            "- 10: Periapical_lesion — Pathological lesions located at the apex of the tooth root.\n"
            "- 11: Root_canal_filling — Material used to fill the cleaned root canal space after endodontic therapy.\n"
            "- 12: Mandibular_alveolar_nerve — The nerve running through the lower jaw, supplying sensation to the teeth and lower lip.\n\n"
            "**yolov8_contents:**\n"
            "A list of detected segmentation entries. Each entry is a list where:\n"
            "- The first element is the class ID (int).\n"
            "- The following four elements are bounding box coordinates: [x1, y1, x2, y2] (float).\n\n"
            "This represents the class label and the top-left (x1, y1) and bottom-right (x2, y2) corners of the bounding box."            
        )
    )
    yolo_results: YoloV8Segmentation
    message: str = Field(
        ..., 
        description="Status or informational message. If successful, result in empty string."
    )


class CvatSegmentation(BaseModel):
    confidence: float = Field(
        ...,
        description="Confidence score of the segmentation prediction, typically between 0 and 1."
    )
    label: str = Field(
        ...,
        description="The class label name for the segmented object."
    )
    type: str = Field(
        ...,
        description="The type of shape, e.g., 'polygon', 'polyline', or 'point'. In this case only 'mask' type exist."
    )
    points: List[int] = Field(
        ...,
        description=(
            "List of points defining the segmentation shape.\n"
            "- For polygons or polylines: a sequence of x, y coordinates (e.g., [x1, y1, x2, y2, ...]).\n"
            "- For points: [rle_points, ...... bbox_x1, bbox_y1, bbox_x2, bbox_y2]. Note that one need to add the mask key for codes in follows\n\n"
            "Example usage:\n\n"
            "```python\n"
            "import base64\n"
            "import requests\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n\n"
            "def show_plot(image_rgb):\n"
            "    plt.imshow(image_rgb)\n"
            "    plt.show()\n\n"
            "def image_to_base64(image_path):\n"
            "    with open(image_path, \"rb\") as image_file:\n"
            "        return base64.b64encode(image_file.read()).decode('utf-8')\n\n"
            "def post_url():\n"
            "    image_path = './tests/files/027107.jpg'\n"
            "    image_base64 = image_to_base64(image_path)\n\n"
            "    data = {\n"
            "        'image': image_base64,\n"
            "        'scale_x': 40/1280,\n"
            "        'scale_y': 30/960,\n"
            "    }\n"
            "    api_proxy_url = 'https://api.smartsurgerytek.net/dentistry-stg/v1/pano_fdi_segmentation_cvat'\n"
            "    api_key = 'please describe your api key here'\n\n"
            "    params = { \n"
            "        'apikey': api_key\n"
            "    }\n\n"
            "    headers = {\n"
            "        'Authorization': f'Bearer {api_key}'  \n"
            "    }\n\n"
            "    response = requests.post(api_proxy_url, json=data, params=params, headers=headers)\n\n"
            "    print(f'Status Code: {response.status_code}')\n"
            "    print(f'Response: {response.json()}')\n"
            "    return response\n\n"
            "def rle2Mask(rle: list[int]) -> np.ndarray:\n"
            "    rle_int_list = list(map(int, rle))\n"
            "    x1, y1, x2, y2 = rle_int_list[-4:]\n"
            "    rle_int_list_filter = rle_int_list[:-4]\n"
            "    width, height = x2 - x1 + 1, y2 - y1 + 1\n\n"
            "    total_pixels = width * height\n"
            "    decoded = np.zeros(total_pixels, dtype=np.uint8)\n"
            "    idx = 0\n"
            "    val = 0\n"
            "    for count in rle_int_list_filter:\n"
            "        count = int(count)\n"
            "        decoded[idx:idx + count] = val\n"
            "        idx += count\n"
            "        val = 1 - val\n\n"
            "    decoded = decoded.reshape((width, height), order='F').T\n"
            "    return decoded\n\n"
            "if __name__ == '__main__':\n"
            "    response = post_url()\n"
            "    results = response.json()\n"
            "    yolov8_contents = results['yolo_results']['yolov8_contents']\n\n"
            "    # example for the first teeth segmentation\n"
            "    cvat_annotation_dict = yolov8_contents[0]\n"
            "    points = yolov8_contents[0]['points']\n"
            "    decoded = rle2Mask(points)\n"
            "    show_plot(decoded)\n"
            "    flattened = decoded.T.flatten(order='F')\n"
            "    cvat_annotation_dict['mask'] = flattened.tolist()\n"
            "    cvat_annotation_dict['frame'] = 0\n"
            "    cvat_annotation_dict['label_id'] = 561\n"
            "```"
        )
    )
    
class CvatSegmentations(BaseModel):
    class_names: Dict[int, str] = Field(
        ...,
        description=(
            "Mapping from class ID to class name and anatomical description:\n\n"
            "- 0: Crown — The visible part of the tooth above the gumline, covered by enamel, responsible for biting and chewing.\n"
            "- 1: Alveolar_bone — The bone structure surrounding and supporting the teeth within the jaw.\n"
            "- 2: Caries — Areas of tooth decay caused by bacterial activity leading to demineralization.\n"
            "- 3: Dentin — The dense, bony tissue beneath the enamel forming the bulk of the tooth.\n"
            "- 4: Pulp — The innermost part of the tooth containing nerves and blood vessels.\n"
            "- 5: Maxillary_sinus — Air-filled spaces located within the bones of the upper jaw near the nose.\n"
            "- 6: Implant — A dental prosthesis surgically inserted into the jawbone to replace missing teeth.\n"
            "- 7: Enamel — The hard, outermost layer of the tooth, protecting it from decay.\n"
            "- 8: Post_and_core — Dental restoration components used to rebuild a tooth's structure after root canal treatment.\n"
            "- 9: Restoration — Materials or procedures used to repair damaged teeth (e.g., fillings, crowns).\n"
            "- 10: Periapical_lesion — Pathological lesions located at the apex of the tooth root.\n"
            "- 11: Root_canal_filling — Material used to fill the cleaned root canal space after endodontic therapy.\n"
            "- 12: Mandibular_alveolar_nerve — The nerve running through the lower jaw, supplying sensation to the teeth and lower lip."
        )
    )
    yolov8_contents: List["CvatSegmentation"] = Field(
        ...,
        description="List of segmentation results, each following the CVAT segmentation format."
    )

class PaSegmentationCvatResponse(BaseModel):
    request_id: int = Field(
        ..., 
        description=(
            "A unique identifier corresponding to the original measurement request. "
            "A value of 0 indicates that the result will not be stored in the database."
        )
    )
    yolo_results: "CvatSegmentations" = Field(
        ..., 
        description="Segmentation results containing CVAT shapes object. Note that in CvatSegmentation, it is necessary to add a mask key by converting the points into a mask."
    )
    message: str = Field(
        ..., 
        description="Status or informational message. If successful, result in empty string."
    )

class PaSegmentationRequest(BaseModel):
    image: str = Field(..., min_length=1,
                        max_length=10_000_000,
                        description='A Base64-encoded periapical (PA) radiographic image, commonly referred to as a bitewing film, used for decompoisition assessment e.g. caries area')  # 增加最大長度限制
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": read_base64_example(),
            }
        }
    )