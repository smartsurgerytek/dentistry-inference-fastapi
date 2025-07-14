import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fastapi import FastAPI, Request, File, APIRouter
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from typing import Annotated
from typing import Any
from typing import Optional
import uvicorn
from src.allocation.service_layer.services import InferenceService
from src.allocation.domain.pa_dental_measure.schemas import PaMeasureDictResponse, ImageResponse, PaMeasureCvatResponse, PaMeasureRequest
from src.allocation.domain.pa_dental_segmentation.schemas import PaSegmentationYoloV8Response, PaSegmentationCvatResponse, PaSegmentationRequest
from src.allocation.domain.pano_caries_detection.schemas import PanoCariesDetectionRequest, PanoCariesDetectionDictResponse
from src.allocation.domain.pano_caries_detection.main import create_pano_caries_detection_model
from src.allocation.domain.pa_pano_classification.main import create_pa_pano_classification_model
from src.allocation.domain.pa_pano_classification.schemas import PaPanoClassificationResponse
from src.allocation.domain.pano_fdi_segmentation.schemas import PanoSegmentationYoloV8Response, PanoSegmentationCvatResponse, PanoSegmentationRequest
from src.allocation.domain.aggregation.schemas import CombinedImageResponse
from src.allocation.domain.leyan_clinic_scenario_classfication.schemas import LeyanClinicScenarioClassificationResponse
from src.allocation.domain.leyan_clinic_scenario_classfication.main import create_Leyan_clinic_scenario_classfication
from contextlib import asynccontextmanager
from ultralytics import YOLO
from src.allocation.adapters.utils import base64_to_bytes
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    #executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 1)
    executor= ThreadPoolExecutor()
    # Load the ML model
    global pa_component_model
    pa_component_model = YOLO('./models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt')

    global pa_component_model2
    pa_component_model2 = YOLO('./models/dentistry_pa-segmentation_yolov11n-seg-all_25.20.pt')

    global pa_contour_model
    pa_contour_model = YOLO('./models/dentistry_pa-contour_yolov11x-seg_25.22.pt')

    global pano_caries_detection_model
    pano_caries_detection_model= create_pano_caries_detection_model(1)

    global pano_caries_detection_model_weight_path
    pano_caries_detection_model_weight_path='./models/dentistry_pano-CariesDetection_resNetFpn_25.12.pth'

    global pa_pano_classification_model
    pa_pano_classification_model=create_pa_pano_classification_model('./models/dentistry_pa-pano-classification_cnn_25.22.pth')

    global leyan_clinic_scenario_classfication_model
    leyan_clinic_scenario_classfication_model = create_Leyan_clinic_scenario_classfication('./models/dentistry_leyan_clinic-classification_cnn_25.28.pth')

    global pano_fdi_segmentation_model
    pano_fdi_segmentation_model= YOLO('./models/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt')
    yield  
    # Cleanup on shutdown
    pa_component_model = None
    pa_contour_model = None
    pano_caries_detection_model= None
    pano_caries_detection_model_weight_path=''
    executor.shutdown(wait=True)

app = FastAPI(
    title="Dental X-ray Inference API",
    version="1.0.0",
    description="API to infer information from dental X-ray images.",
    lifespan=lifespan,
    #docs_url=None,
    #redoc_url=None,
)

v1_router = APIRouter()
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "message": "pydantic model validation failed!",
            "details": [
                {
                    "loc": err["loc"],
                    "msg": err["msg"],
                    "type": err["type"]
                }
                for err in exc.errors()
            ]
        }
    )

@app.get("/", response_model=str)
async def read_root() -> str:
    return "Welcome to Smart Surgery Dentistry APIs! "

@v1_router.post("/pa_measure_dict", response_model=PaMeasureDictResponse)
async def generate_periapical_film_measure_dict(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest,
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_dict(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)

@v1_router.post("/pa_measure_cvat", response_model=PaMeasureCvatResponse)
async def generate_periapical_film_measure_dict(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest,
) -> PaMeasureDictResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_cvat(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)
@v1_router.post("/pa_measure_image", response_model=ImageResponse)#, response_model=DentalMeasureDictResponse)
async def generate_periapical_film_measure_image_base64(
    # image: str,
    # #scale: Any, #: expected Annotated[str, Form()] or array
    # scale_x: float,
    # scale_y: float,  
    request: PaMeasureRequest
) -> ImageResponse:
    #scale_obj=ScaleValidator(scale=scale)
    image=base64_to_bytes(request.image)
    return InferenceService.pa_measure_image_base64(image, pa_component_model, pa_contour_model, request.scale_x, request.scale_y)

@v1_router.post("/pa_segmentation_yolov8", response_model=PaSegmentationYoloV8Response)
async def generate_periapical_film_segmentations_yolov8(
    #image: str,
    request: PaSegmentationRequest,
) -> PaSegmentationYoloV8Response:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_yolov8(image, pa_component_model, pa_component_model2)

@v1_router.post("/pa_segmentation_cvat", response_model=PaSegmentationCvatResponse)
async def generate_periapical_film_segmentations_cvat(
    #image: str,
    request: PaSegmentationRequest
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_cvat(image, pa_component_model, pa_component_model2)

@v1_router.post("/pa_segmentation_image", response_model=ImageResponse)
async def generate_periapical_film_segmentations_image_base64(
    request: PaSegmentationRequest
) -> PaSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_segmentation_image_base64(image, pa_component_model, pa_component_model2)

@v1_router.post("/pano_caries_detection_image", response_model=ImageResponse)
async def generate_pano_caries_detection_image_base64(
    request: PanoCariesDetectionRequest
) -> ImageResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_caries_detection_image_base64(image, pano_caries_detection_model, pano_caries_detection_model_weight_path)

@v1_router.post("/pano_caries_detection_dict", response_model=PanoCariesDetectionDictResponse)
async def generate_pano_caries_detection_dict(
    request: PanoCariesDetectionRequest
) -> PanoCariesDetectionDictResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_caries_detection_dict(image, pano_caries_detection_model, pano_caries_detection_model_weight_path)

@v1_router.post("/pa_pano_classification_dict", response_model=PaPanoClassificationResponse)
async def generate_pa_pano_classification(
    request: PaSegmentationRequest
) -> PaPanoClassificationResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pa_pano_classification_dict(image, pa_pano_classification_model)


@v1_router.post(
    "/leyan_clinic_scenario_classification_dict", 
    response_model=LeyanClinicScenarioClassificationResponse  # 規則一：給 FastAPI 的規則
)
async def generate_leyan_clinic_scenario_classification(
    request: PaSegmentationRequest
) -> LeyanClinicScenarioClassificationResponse: # 規則二：給開發者看的「說明書」
    """
    接收一張圖片，並回傳 Leyan 臨床情境的分類結果。
    """
    image_bytes = base64_to_bytes(request.image)
    
    # 這裡回傳的是 LeyanClinicScenarioClassificationResponse 物件
    result = InferenceService.leyan_clinic_scenario_classification_dict(
        image=image_bytes, 
        model=leyan_clinic_scenario_classfication_model
    )
    
    return result # 規則三：實際回傳的內容


@v1_router.post("/pano_fdi_segmentation_yolov8", response_model=PanoSegmentationYoloV8Response)
async def generate_fdi_panoramic_xray_segmentations_yolov8(
    #image: str,
    request: PanoSegmentationRequest,
) -> PanoSegmentationYoloV8Response:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_fdi_segmentation_yolov8(image, pano_fdi_segmentation_model)

@v1_router.post("/pano_fdi_segmentation_cvat", response_model=PanoSegmentationCvatResponse)
async def generate_fdi_panoramic_xray_segmentations_cvat(
    #image: str,
    request: PanoSegmentationRequest
) -> PanoSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_fdi_segmentation_cvat(image, pano_fdi_segmentation_model)

@v1_router.post("/pano_fdi_segmentation_image", response_model=ImageResponse)
async def generate_fdi_panoramic_xray_segmentations_image_base64(
    request: PanoSegmentationRequest
) -> PanoSegmentationCvatResponse:
    image=base64_to_bytes(request.image)
    return InferenceService.pano_fdi_segmentation_image_base64(image, pano_fdi_segmentation_model)

@v1_router.post("/pa_aggregation_images", response_model=CombinedImageResponse)
async def aggregate_pa_images_base64_dict(
    request: PaMeasureRequest
) -> CombinedImageResponse:
    
    image = base64_to_bytes(request.image)
    loop = asyncio.get_running_loop()

    # 定義所有需要執行的推論任務
    tasks = [
        loop.run_in_executor(
            executor,
            InferenceService.pa_measure_image_base64,
            image, pa_component_model, pa_contour_model,
            request.scale_x, request.scale_y
        ),
        loop.run_in_executor(
            executor,
            InferenceService.pa_segmentation_image_base64,
            image, pa_component_model, pa_component_model2
        ),
        # 你可以在這裡繼續加入更多任務
        # loop.run_in_executor(executor, your_other_inference_fn, image, ...)
    ]

    # 並行等待所有推論結果
    results = await asyncio.gather(*tasks)

    # 拆解結果（根據順序）
    measure_img = results[0]
    segmentation_img = results[1]

    return CombinedImageResponse(
        measure_image_base64=measure_img.image,
        segmentation_image_base64=segmentation_img.image
    )


app.include_router(v1_router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app)
    #uvicorn.run("src.allocation.entrypoints.fast_api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
    #uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))