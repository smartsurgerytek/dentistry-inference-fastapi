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
## cvat install

docker installation
```
sudo apt-get update
sudo apt-get --no-install-recommends install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg-agent \
  software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"
sudo apt-get update
sudo apt-get --no-install-recommends install -y \
  docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

recommend to clone by ssh
```
git clone https://github.com/cvat-ai/cvat.git

or

git clone git@github.com:cvat-ai/cvat.git
```

export cvat host
```
export CVAT_HOST=FQDN_or_YOUR-IP-ADDRESS
```

run docker compose up
```
docker compose up -d
```

create super user
```
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

## CVAT auto annotation and Nuclio deploy

cvat docker compose up

```
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d
```

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

create project 
```
nuctl create project cvat
```

run deploy SAM
```
./serverless/deploy_gpu.sh /serverless/pytorch/facebookresearch/sam/nuclio
or
nuctl deploy --project-name cvat --path "/serverless/pytorch/facebookresearch/sam/nuclio" --platform local
```


# add the huggingface token
```
echo "huggingface token" > ./conf/hf_token.txt
```

Deploy the functions in nuclio
```
nuctl deploy --project-name cvat --path "./src/allocation/service_layer/cvat_nuclio/segmentation_PA" --platform local
```

```
nuctl deploy --project-name cvat --path "./src/allocation/service_layer/cvat_nuclio/segmentation_PANO" --platform local
```

check functions aviability in nuclio
```
nuctl get functions
```

## gcloud run depoly

docker build (convension 2025 year, 12th weeks)

```
docker build -t dentistry-inference-core-2512
```

docker run for test
```
docker run -p 8080:8080 dentistry-inference-core-2512
```

docker push (recommnad this way, otherwise one need to build image whole time when depolying)


```
docker tag dentistry-inference-core-2512 <your docker account name>/dentistry-inference-core-2512:latest
docker push <your docker account name>/dentistry-inference-core-2512:latest
```

gcloud run depoly

```
gcloud run deploy dentistry-inference-core-2512 --image <your docker account name>/dentistry-inference-core-2512:latest --cpu=8 --memory=16Gi --region asia-east1 --platform managed
```

## Apigee proxy work flow

1. go to the apigee console, logging and choose the sandbox (ex: sandbox-446907)
apigee.google.com

2. Deveop API Proxies: create a new proxy (for example: smartsurgery-dentistry)

choose the reverse proxy and setting the name, path, target (target should be gcloud run url entry)

3. Publish API Products: edit the product

deploy the proxy

4. Publish Developers: create a new developer

5. Publish Apps: create a new app and the API key will be generated (for example: QpDPpMYkSAFJd0RNFI2eU15Ri5aA7ePWqTk4jhkr4c2mTzn9)

6. Admin Environments Groups: check the hostnames (for example 34.107.237.238.nip.io)

7. test the proxy:

for eaxmple, https://34.107.237.238.nip.io/smartsurgery-dentistry?apikey=QpDPpMYkSAFJd0RNFI2eU15Ri5aA7ePWqTk4jhkr4c2mTzn9

please check whether it is same as the gcloud run url

# create the SSL cerification with google

gcloud apigee envgroup-ssl-certs create dentistry-cert-apigee-managed \
  --envgroup=default-envgroup \
  --type=managed \
  --domains=api.smartsurgerytek.net \
  --project=staging-456206

## create the SSL cerification with Let's Encrypt

see the jira card
https://smartsurgerytek.atlassian.net/browse/MSA-363


Manually upload that
```
sudo certbot certonly --manual --preferred-challenges dns -d api.smartsurgerytek.net
```

# SaSS schedulely set the minimal instance
uptime 16 hours (8am-12am)
## first setup (windows commnad)
create topic 
```
gcloud pubsub topics create cloud-builds
```

create bucket
```
gcloud storage buckets create smartsurgery-dentistry-scheduler-stroage --location=asia-east1
```

set the cloud build yaml set-min-instances-0.yaml (as shown in ./config/set-min-instances-0.yaml)

also set up the set-min-instances-1.yaml
```
steps:
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'services'
      - 'update'
      - 'dentistry-inference-core-2514'
      - '--min-instances=0'
      - '--region=asia-east1'
```

scheduler command
```
gcloud scheduler jobs create pubsub set-dentistry-min-instance ^
  --schedule="0 8 * * 1-5" ^
  --time-zone="Asia/Taipei" ^
  --location=asia-east1 ^
  --topic=cloud-builds ^
  --message-body="{\"build\": {\"source\": {\"storageSource\": {\"bucket\": \"smartsurgery-dentistry-scheduler-stroage\", \"object\": \"set-min-instances-1.yaml\"}}}}"

gcloud scheduler jobs create pubsub disable-dentistry-min-instance ^
  --schedule="0 0 * * 2-6" ^
  --time-zone="Asia/Taipei" ^
  --location=asia-east1 ^
  --topic=cloud-builds ^
  --message-body="{\"build\": {\"source\": {\"storageSource\": {\"bucket\": \"smartsurgery-dentistry-scheduler-stroage\", \"object\": \"set-min-instances-1.yaml\"}}}}"

```
