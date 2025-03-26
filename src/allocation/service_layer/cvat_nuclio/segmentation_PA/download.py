import io, base64, cv2, json
import numpy as np
from ultralytics import YOLO

from src.allocation.domain.pa_dental_segmentation.main import *
from src.allocation.service_layer.cvat_nuclio.segmentation_PA.download import down_load_function

def init_context(context):
    context.logger.info("Init context...  0%")
    down_load_function()
    context.user_data.model_handler = YOLO('/opt/nuclio/models/dentistry_yolov11x-seg-all_4.42.pt')
    context.logger.info("Init context...100%")
    
def handler(context, event):
    context.logger.info("Run sst model")
    data = event.body
    image_bytes = io.BytesIO(base64.b64decode(data["image"]))
    nparr  = np.frombuffer(image_bytes.getvalue(),  np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result_dict=yolo_transform(image, context.user_data.model_handler, return_type='cvat_mask', plot_config=None, tolerance=0.5)
    rt=result_dict['yolov8_contents']
    return context.Response(body=json.dumps(rt), headers={},
        content_type='application/json', status_code=200)