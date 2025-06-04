import io, base64, cv2, json
import numpy as np
from ultralytics import YOLO
from skimage.measure import find_contours, approximate_polygon

import os 
import sys

from src.allocation.domain.pa_dental_segmentation.main import *
from src.allocation.service_layer.download import donw_load_function

def init_context(context):
    context.logger.info("Init context...  0%")
    os.makedirs('/opt/nuclio/models', exist_ok=True)
    donw_load_function('/opt/nuclio/models')
    context.user_data.model_handler = YOLO('/opt/nuclio/models/dentistry_pa-segmentation_yolov11n-seg-all_25.20.pt')
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


# if __name__ == '__main__':
#     model=YOLO('./models/dentistry_yolov11x-seg-all_4.42.pt')
#     image=cv2.imread('./tests/files/nomal-x-ray-0.8510638-270-740_0_2022011008.png')
#     with open('./conf/dentistry_PA.yaml', 'r') as file:
#         config=yaml.safe_load(file)
#     test1=yolo_transform(image, model, return_type='cvat_mask', plot_config=None, tolerance=0.5)
#     rt = transform(image, model)
#     breakpoint()    