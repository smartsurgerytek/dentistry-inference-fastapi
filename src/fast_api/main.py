from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import uvicorn
import json
import zlib
import os
import sys
current_directory = os.path.dirname(__file__)
model_inference_directory = os.path.abspath(os.path.join(current_directory, '../model_inference'))
sys.path.insert(0, model_inference_directory)
from retrieve_yolo_results import retrieve_yolo_results

# Initialize FastAPI
app = FastAPI()

# load plot setting
with open('./conf/mask_color_setting.json', 'r') as file:
    plot_setting = json.load(file)

# Load the YOLO model
model = YOLO('./dentistry_yolov8n_20240807_all.pt')  # Replace with your model path


@app.get("/", response_model=str)
async def read_root() -> str:
    """返回一個歡迎消息"""
    return "Welcome to Smart Surgery Dentistry APIs!"

@app.post("/inference_return_list/")
async def inference_result(file: UploadFile = File(...)):

    try:
        # Read the image file
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        # Perform inference
        results = model(image)
        result=results[0]
        detections=retrieve_yolo_results(result, image, plot_setting, return_type='dict')

        return detections
    
    except Exception as e:
        # Log the exception
        print(f"Error processing file: {e}")
        return JSONResponse(content={"error": "Failed to process the image"}, status_code=500)

@app.post("/inference_plot/")
async def inference_plot(file: UploadFile = File(...)):

    # Read the image file
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_plot=image.copy()
    if image is None:
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)
    # Perform inference
    results = model(image)
    result=results[0]
    image_plot=retrieve_yolo_results(result, image_plot, plot_setting, return_type='plot')

    if image_plot is None:
        return JSONResponse(content={"error": "No mask detected"}, status_code=400)
    
    _, buffer = cv2.imencode('.png', image_plot)
    return StreamingResponse(BytesIO(buffer), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app)