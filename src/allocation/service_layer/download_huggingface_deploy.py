from huggingface_hub import hf_hub_download
import shutil
import os
import yaml



def donw_load_function(base_path):
    hf_token=None
    if os.environ.get("HUGGINGFACE_TOKEN"):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token is None:
        with open('./conf/credential.yaml', 'r', encoding='utf-8') as file:
            credentials = yaml.safe_load(file)
        hf_token = credentials['HUGGINGFACE_TOKEN']
        if hf_token=="please write the token here":
            raise ValueError('Please write the token in credential.yaml or set HUGGINGFACE_TOKEN as env variable')
        
    print('hf_token', hf_token)    

    repo_id = "smartsurgery/dentistry-models"  # repo 名稱
    os.makedirs(base_path,exist_ok=True)
    model_path = [
        "PA_dental_contour/dentistry_pa-contour_yolov11x-seg_25.22.pt",
        "PA_segmentation/dentistry_pa-segmentation_yolov11x-seg-all_24.42.pt",
        "PA_segmentation/dentistry_pa-segmentation_yolov11n-seg-all_25.20.pt",
        "PANO_caries_detection/dentistry_pano-CariesDetection_resNetFpn_25.12.pth",
        "PANO_fdi_segmentation/dentistry_pano-fdi-segmentation_yolo11x-seg_25.12.pt",
    ]

    save_map = {
        key: os.path.join(base_path , key.split("/")[-1])
        for key in model_path
    }
    print(save_map)
    for filename, save_path in save_map.items():
        if os.path.exists(save_path):
            continue
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        print(f"Downloading {filename} to {save_path}")
        shutil.copy(file_path, save_path)

if __name__ == "__main__":
    base_path = "./models"
    donw_load_function(base_path)