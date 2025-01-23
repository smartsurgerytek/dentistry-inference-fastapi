# Dentistry-inference-core
This repo is developed with dentistry-related functions such as dental segmentation, periodontal estimation, Caries detection.

## Introduction
See the details of Proper Nouns definition in dentistry and introduce the desirable features and works.

[See the introduction](./docs/introduction.md)

## Github workflow and Coding Convention

In general, we follow the Google coding style guide and Github feature branch workflow.

[See the coding convention](./docs/coding_convention.md)

## Paper supports
One can follow the previous work, no need to build everything from zeros.

[Read on papers](./docs/papers.md)


## Src Files Introduction
The function created for periodontal estimation currently run with two ML models: dental_segmentation and contour_detection
[Dental_measure](./src/allocation/domain/dental_measure/main.py)

The function created for segmentation task and outputs yolov8 annotation format
[Dental_segmentation](./src/allocation/domain/dental_segmentation/main.py)

## Download models from Huggingface

```
huggingface-cli login
```
or

```
huggingface-cli login --token <your token> --add-to-git-credential
```

After loggin with token, run the download scipt

```
python src/allocation/service_layer/download.py
```

## Run performance test

(If user don't want to run performance test, just remove tests/performance/)

After downloading models from huggingface,

download split_data_4.42.rar dataset from googlecloud
```
https://drive.google.com/drive/u/2/folders/1pVjfbgGWWcPv0x4HVd1HNvlm8Fwi5VNg
```
unrar to ./datasets/split_data_4.42

Setting config to absolute path for path in ./config/dentistry.yaml

ex:

path: D:\boan\repo\smart_surgery_repo\Dentistry-Inference-PeriodontalDisease\datasets\split_data_4.42

## Pytest command
```
pytest -s -vv
```
See the coverage

```
pytest -vv --cov src/
```

## CVAT-nuclio deploy

clone the repo in CVAT folder
```
git clone https://github.com/smartsurgerytek/dentistry-inference-core.git
```

Install nuclio
```
curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
			| grep -i "browser_download_url.*nuctl.*$(uname)" \
			| cut -d : -f 2,3 \
			| tr -d \" \
			| wget -O nuctl -qi - && chmod +x nuctl
```

Deploy the functions in nuclio
```
nuctl deploy --project-name cvat --path "./dentistry-inference-core/src/allocation/service_layer/cvat_nuclio/segmentation_PA" --platform local
```

```
nuctl deploy --project-name cvat --path "./dentistry-inference-core/src/allocation/service_layer/cvat_nuclio/segmentation_PANO" --platform local
```

check functions aviability in nuclio
```
nuctl get functions
```