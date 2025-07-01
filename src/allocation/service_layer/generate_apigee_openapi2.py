import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from fastapi_swagger2 import FastAPISwagger2
from src.allocation.entrypoints.fast_api import app
import yaml

FastAPISwagger2(app)

URL='https://api.smartsurgerytek.net/dentistry-stg/'
app.servers.append(URL)
spec = app.swagger2()
spec['x-google-backend'] = {'address': URL}

##drop some items
for key1, def_content in spec['definitions'].items():
    if spec['definitions'].get('properties') is None:
        continue
    for key2, property_content in def_content['properties'].items():
        if 'prefixItems' in property_content.keys():
            del spec['definitions'][key1]['properties'][key2]['prefixItems']
del spec['definitions']['YoloV8Segmentation']['properties']['yolov8_contents']['items']['items']

security_definitions = {
    "securityDefinitions": {
        "api_key": {
            "type": "apiKey",
            "name": "apikey", # apigee: apikey ; google gateway: key 
            "in": "query"
        }
    }
}
spec.update(security_definitions)

###if one want to make it sepecific, insert it separetely
security = {
    "security": [
        {
            "api_key": [] 
        }
    ]
}
spec.update(security)

with open('./conf/apigee_openai2_spec.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(spec, file, sort_keys=False, allow_unicode=True)
