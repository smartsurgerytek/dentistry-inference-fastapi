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
[Dental_measure](./src/Dental_measure/main.py)

The function created for segmentation task and outputs yolov8 annotation format
[Dental_segmentation](./src/Dental_segmentation/main.py)

## Download models from Huggingface

```
huggingface-cli login
```

After loggin with token, run the download scipt

```
python src/allocation/service_layer/download.py
```

## Pytest command
```
pytest -vv
```
See the coverage

```
pytest -vv --cov src/
```