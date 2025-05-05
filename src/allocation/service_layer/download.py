from huggingface_hub import hf_hub_download
import shutil
import os
repo_id = "smartsurgery/dentistry-models"  # repo 名稱

hf_token=None
if os.environ.get("HUGGINGFACE_TOKEN"):
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

if hf_token is None:
    with open('./conf/hf_token.txt', 'r', encoding='utf-8') as file:
        hf_token = file.read()
    print(hf_token)

os.makedirs('./models',exist_ok=True)
save_map={
    "PA_dental_contour/dentistry_pa-contour_yolov11n-seg_24.46.pt":"./models/dentistry_pa-contour_yolov11n-seg_24.46.pt",
    "PA_segmentation/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt":"./models/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt",
    "PANO_caries_detection/dentistry_pano-CariesDetection_resNetFpn_25.12.pth":"models/dentistry_pano-CariesDetection_resNetFpn_25.12.pth",
    "PA_PANO_classification/dentistry_pa-pano-classification_cnn_25.10.pth":"models/dentistry_pa-pano-classification_cnn_25.10.pth",
    "PANO_fdi_segmentation/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt":'models/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt'
    }

for filename, save_path in save_map.items():
    if os.path.exists(save_path):
        continue
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
    shutil.copy(file_path, save_path)
