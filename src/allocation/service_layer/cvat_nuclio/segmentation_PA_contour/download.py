from huggingface_hub import hf_hub_download
import shutil
import os
from src.allocation.service_layer.download import model_path
with open('./conf/hf_token.txt', 'r', encoding='utf-8') as file:
    hf_token = file.read().strip()
    print(hf_token)

def donw_load_function():
    repo_id = "smartsurgery/dentistry-models"  # repo 名稱

    os.makedirs('./models',exist_ok=True)
    base_path = "/opt/nuclio/models/"
    save_map = {
    key: base_path + key.split("/")[-1]
    for key in model_path
    }

    for filename, save_path in save_map.items():
        if os.path.exists(save_path):
            continue
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        shutil.copy(file_path, save_path)
