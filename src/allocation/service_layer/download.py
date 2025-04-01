from huggingface_hub import hf_hub_download
import shutil
import os
repo_id = "smartsurgery/dentistry-models"  # repo 名稱
filename = "dental_contour/dentistry_yolov11n-seg_contour4.46.pt"

os.makedirs('./models',exist_ok=True)
save_map={
    "PA_dental_contour/dentistryContour_yolov11n-seg_4.46.pt":"./models/dentistryContour_yolov11n-seg_4.46.pt",
    "PA_segmentation/dentistry_yolov11x-seg-all_4.42.pt":"./models/dentistry_yolov11x-seg-all_4.42.pt",
    "PANO_caries_detection/dentistry_pano-caries-detection-resNetFpn_5.12.pth":"models/dentistry_pano-caries-detection-resNetFpn_5.12.pth",
    "PA_PANO_classification/pa_pano_classification.pth":"models/pa_pano_classification.pth",
    "PANO_fdi_segmentation/pano_fdi_segmentation_25.12.pt":'models/pano_fdi_segmentation_25.12.pt'
    }

for filename, save_path in save_map.items():
    if os.path.exists(save_path):
        continue
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy(file_path, save_path)
