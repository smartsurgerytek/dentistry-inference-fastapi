from huggingface_hub import hf_hub_download
import shutil
import os
repo_id = "smartsurgery/dentistry-models"  # repo 名稱
filename = "dental_contour/dentistry_yolov11n-seg_contour4.46.pt"

os.makedirs('./models',exist_ok=True)
save_map={
    "dental_contour/dentistryContour_yolov11n_4.46.pt":"./models/dentistryContour_yolov11n_4.46.pt",
    "all_category/dentistry_yolov11x-seg_4.42.pt":"./models/dentistry_yolov11x-seg_4.42.pt"}
for filename, save_path in save_map.items():
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy(file_path, save_path)
print(file_path)