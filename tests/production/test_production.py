import base64
import requests
import os
import sys
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.allocation.entrypoints.fast_api import app
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

api_key = None
if os.environ.get("DENTISTRY_API_KEY"):
    api_key = os.environ.get("DENTISTRY_API_KEY")

if api_key is None:
    with open('./conf/credential.yaml', 'r', encoding='utf-8') as file:
        credentials = yaml.safe_load(file)
    api_key = credentials['DENTISTRY_API_KEY']
    if api_key=="please write the token here":
        raise ValueError('Please write the token in credential.yaml or set DENTISTRY_API_KEY as env variable')

if api_key is None:
    raise Exception("DENTISTRY_API_KEY is not set")

api_domain_url = 'https://api.smartsurgerytek.net/dentistry-stg'
#api_domain_url= 'http://127.0.0.1:8000'

params = { 
    'apikey': api_key
}
headers = {
    'Authorization': f'Bearer {api_key}'  
}

def test_production():
    pa_image_path = './tests/files/caries-0.6741573-260-760_1_2022052768.png'
    pa_image = image_to_base64(pa_image_path)
    pano_image_path = './tests/files/027107.jpg'
    pano_image = image_to_base64(pano_image_path)

    all_path_list=[element.path for element in app.routes]

    for path in all_path_list:
        if 'pa' in path:
            image=pa_image
        elif 'pano' in path:
            image=pano_image
        else:
            continue

        data = {
            'image': image,
            'scale_x': 31/1280,
            'scale_y': 31/960,
        }

        api_proxy_url=api_domain_url+path

        response = requests.post(api_proxy_url, 
                                    json=data, 
                                    params=params, 
                                    headers=headers
                                    )
        print(path)
        print(f'Status Code: {response.status_code}')
        #print(f'Response: {response.json()}')

        assert response.status_code == 200, f"❌ API 回傳錯誤: {response.status_code}, Response: {response.text}, path: {path}"
    
def test_production_black_image():
    black_image_path = './tests/files/black.png'
    image = image_to_base64(black_image_path)
    all_path_list=[element.path for element in app.routes]

    for path in all_path_list:
        if 'pa' in path:
            pass
        elif 'pano' in path:
            pass
        else:
            continue

        data = {
            'image': image,
            'scale_x': 31/1280,
            'scale_y': 31/960,
        }

        api_proxy_url=api_domain_url+path

        response = requests.post(api_proxy_url, 
                                    json=data, 
                                    params=params, 
                                    headers=headers
                                    )
        print(path)
        print(f'Status Code: {response.status_code}')
        #print(f'Response: {response.json()}')

        assert response.status_code == 200, f"❌ API 回傳錯誤: {response.status_code}, Response: {response.text}, path: {path}"

# if __name__ == '__main__':
#     test_production_black_image()