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
# Initialize FastAPI
app = FastAPI()

# load plot setting
with open('./conf/mask_color_setting.json', 'r') as file:
    color_dict = json.load(file)

plot_setting={
    'color_dict':color_dict,
    'font_face':cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale':1,
    'thickness':3,
    'line_type':cv2.LINE_AA,
}

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
        
        if results is None or len(results) == 0:
            return []

        detections = []
        for result in results:
            # Check type of result
            if not hasattr(result, 'names') or not hasattr(result, 'boxes'):
                continue
            
            # 獲取類別名稱
            class_names = result.names    
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            
            if masks is not None:
                for mask, box in zip(masks.data, boxes):
                    # Get class ID and confidence
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # Get class name
                    class_name = class_names[class_id]

                    # Convert mask to numpy array and resize to match original image
                    mask_np = mask.cpu().numpy()
                    mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                    
                    # Convert mask to binary image
                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                    #compressed_mask = zlib.compress(mask_binary.tobytes())
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "segmentation_mask": mask_binary.tolist()
                    })
        return detections
    
    except Exception as e:
        # Log the exception
        print(f"Error processing file: {e}")
        return JSONResponse(content={"error": "Failed to process the image"}, status_code=500)

@app.post("/inference_plot/")
async def inference_plot(file: UploadFile = File(...)):

    font_face = plot_setting['font_face']  # Font type
    font_scale = plot_setting['font_scale']  # Font size
    thickness = plot_setting['thickness']  # Text thickness
    line_type = plot_setting['line_type']  # Line type
    color_dict= plot_setting['color_dict']
    # Read the image file
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    plot_image=image.copy()
    if image is None:
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

    # Perform inference
    results = model(image)
    
    for result in results:
        # 獲取類別名稱
        class_names = result.names
        class_name_list=[]        
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        
        if masks is not None:
            for mask, box in zip(masks.data, boxes):
                # Get class ID and confidence
                class_id = int(box.cls)
                #confidence = float(box.conf)
                
                # Get class name
                class_name = class_names[class_id]

                # Convert mask to numpy array and resize to match original image
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                
                # Convert mask to binary image
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                
                # Check if the mask is valid
                if np.sum(mask_binary) == 0:
                    continue
                
                # Apply mask to the original image
                mask_colored = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)

                if class_name != 'Background':# background label is no needing
                    mask_colored[mask_binary == 255] = color_dict[str(class_id)]
                    
                    # Overlay the colored mask
                    plot_image = cv2.addWeighted(plot_image, 1, mask_colored, 0.8, 0)
                    
                    # Calculate the average color of the masked area
                    map_color = (plot_image[mask_binary == 255].sum(axis=0) / plot_image[mask_binary == 255].shape[0]).astype(np.uint8).tolist()
                    
                    # Add class name to the list if not already present
                    if class_name not in class_name_list:
                        class_name_list.append(class_name)
                        
                        # Draw class name on the image
                        text_position = (mask_binary.shape[0] // 20, mask_binary.shape[1] // 10 * (len(class_name_list)))
                        cv2.putText(plot_image, class_name, text_position, font_face, font_scale, map_color, thickness, line_type)
        else:
            return JSONResponse(content={"error": "No mask detected"}, status_code=400)
    _, buffer = cv2.imencode('.png', plot_image)
    return StreamingResponse(BytesIO(buffer), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app)