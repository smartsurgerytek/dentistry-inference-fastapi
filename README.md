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

2. Deveope API Proxies: create a new proxy (for example: smartsurgery-dentistry). Note that only inmediate proxy surpport the apikey verification.

choose the reverse proxy and setting the name, path, target (target should be gcloud run url entry)

If apikey is required, please go to the develop page and create a "verify API key" policy and add it under Proxy endpoint: PreFlow in default.

3. Publish API Products: edit the product

deploy the proxy

4. Publish Developers: create a new developer

5. Publish Apps: create a new app and the API key will be generated (for example: QpDPpMYkSAFJd0RNFI2eU15Ri5aA7ePWqTk4jhkr4c2mTzn9)

6. Check IP and hostname: go to environments groups check the hostnames and check ip in the apigee load balancer.

7. SSL certification: create one google certificate and setting in the apigee load balancer if using self-defined domain. (Optional)

8. test the proxy: (the above step may requires, one can check after one hour)

for eaxmple, https://34.107.237.238.nip.io/smartsurgery-dentistry?apikey=QpDPpMYkSAFJd0RNFI2eU15Ri5aA7ePWqTk4jhkr4c2mTzn9

please check whether it is same as the gcloud run url

## Apigee portals set up

1. Add the CORS policy in Deveope pages and put in under Target endpoint:PreFlow in default.

2. Goes to the portal page and crate protal pages.

3. using my code src/allocation/service_layer/generate_apigee_openapi2.py to generate the openapi2 yaml

4. Upload in API Catalog

5. View live portal and test with API key

# create the SSL cerification with google

gcloud apigee envgroup-ssl-certs create dentistry-cert-apigee-managed \
  --envgroup=default-envgroup \
  --type=managed \
  --domains=api.smartsurgerytek.net \
  --project=staging-456206

Note that the command is unable to use in 20250630, one need to manully create the SSL certificate
## create the SSL cerification with Let's Encrypt

see the jira card
https://smartsurgerytek.atlassian.net/browse/MSA-363


Manually upload that
```
sudo certbot certonly --manual --preferred-challenges dns -d api.smartsurgerytek.net
```

# SaSS schedulely set the minimal instance
uptime 16 hours (8am-12am)
## first setup (windows command)
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


# Issue tracking: PORT=8080 issue and timeout
```
ERROR: (gcloud.run.deploy) Revision 'dentistry-inference-core-2514-00035-9b6' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.
```
there are serveral reasons for this error.

one can check the jira card

https://smartsurgerytek.atlassian.net/browse/MSA-454

1. too small device: increasing the cpu and memory might solve the problem
2. port setting issue: check the port whether is 8080 in entrypoints 
3. cold start too long: one need to set up the startup-probe
for example:
```
gcloud run deploy dentistry-inference-core-2514 \
  --image=asia-east1-docker.pkg.dev/sandbox-446907/inference-core/dentistry-inference-core-2514 \
  --cpu=8 \
  --memory=16Gi \
  --region=asia-east1 \
  --platform=managed \
  --timeout=900s \
  --port=8080 \
  --startup-probe=tcpSocket.port=8080,initialDelaySeconds=0,timeoutSeconds=240,failureThreshold=1,periodSeconds=240
```

## find the log
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="dentistry-inference-core-2514"' \
  --project=sandbox-446907 \
  --limit=100 \
  --format="value(textPayload)"

# issue tracking: ERROR: (gcloud.run.deploy) Revision 'dentistry-inference-core-2514-00100-6mh' is not ready and cannot serve traffic. Container import failed.

one can build the image directly to check
```
# 1. build image
docker build --build-arg HUGGINGFACE_TOKEN=<huggingface_token> -t asia-east1-docker.pkg.dev/sandbox-446907/cloud-run-source-deploy/dentistry-inference-core-2514 .

# 2. push image to Artifact Registry
docker push asia-east1-docker.pkg.dev/sandbox-446907/cloud-run-source-deploy/dentistry-inference-core-2514

# 3. (optional) local test
docker run -e PORT=8080 -p 8080:8080 asia-east1-docker.pkg.dev/sandbox-446907/cloud-run-source-deploy/dentistry-inference-core-2514

# 4. deploy to Cloud Run
gcloud run deploy dentistry-inference-core-2514 \
  --image=asia-east1-docker.pkg.dev/sandbox-446907/cloud-run-source-deploy/dentistry-inference-core-2514 \
  --cpu=8 \
  --memory=32Gi \
  --region=asia-east1 \
  --platform=managed \
  --timeout=900s \
  --port=8080 \
  --startup-probe=tcpSocket.port=8080,initialDelaySeconds=30,timeoutSeconds=240,failureThreshold=3,periodSeconds=240
```

```
Deploying container to Cloud Run service [dentistry-inference-core-2514] in project [sandbox-446907] region [asia-east1]
X Deploying...                                                                                                                                                                                 
  - Creating Revision...                                                                                                                                                                       
Deployment failed                                                                                                                                                                              
ERROR: (gcloud.run.deploy) Revision 'dentistry-inference-core-2514-00100-6mh' is not ready and cannot serve traffic. Container import failed.
```

## Manually delete problematic one

```
gcloud run revisions list --service=dentistry-inference-core --region=asia-east1
```

for example
```
   REVISION                                 ACTIVE  SERVICE                        DEPLOYED                 DEPLOYED BY
X  dentistry-inference-core-2514-00131-tkd          dentistry-inference-core-2514  2025-06-28 11:45:32 UTC  298229070754-compute@developer.gserviceaccount.com
✔  dentistry-inference-core-2514-00130-zfr  yes     dentistry-inference-core-2514  2025-06-28 06:11:12 UTC  298229070754-compute@developer.gserviceaccount.com
✔  dentistry-inference-core-2514-00129-t7x  yes     dentistry-inference-core-2514  2025-06-20 09:17:08 UTC  298229070754
```

```
gcloud run services update-traffic dentistry-inference-core-2514 ^
  --to-revisions=dentistry-inference-core-2514-00130-zfr=100 ^
  --region=asia-east1
```

```
gcloud run revisions delete dentistry-inference-core-2514-00131-tkd --region=asia-east1
```

## list aviailable revisions
```
gcloud run revisions list --service=dentistry-inference-core-2514 --region=asia-east1
```
one need to manually delete problematic one

## delete problematic revision

cloud run -> dentistry-inference-core-2514 -> revisions -> dentistry-inference-core-2514-00100-6mh -> Action: delete

## robot tests

```
robot tests/production/test_production_detail.robot
```
Once the tests have run successfully, you should see output similar to the following in your terminal:
```
==============================================================================
Test Production Detail
==============================================================================
Test PA Aggregation Images - Valid                                    
Test PA Aggregation Images - Valid                                    | PASS |
------------------------------------------------------------------------------
Test PA Aggregation Images - Empty                                    
Test PA Aggregation Images - Empty                                    | PASS |
------------------------------------------------------------------------------
Test PA Aggregation Images - Malformed                                
Test PA Aggregation Images - Malformed                                | PASS |
------------------------------------------------------------------------------
                                      .
                                      .
                                      .
                                      .
                                      .         
------------------------------------------------------------------------------
Test Pano FDI Segmentation YOLOv8 - Empty                             
Test Pano FDI Segmentation YOLOv8 - Empty                             | PASS |
------------------------------------------------------------------------------
Test Pano FDI Segmentation YOLOv8 - Malformed                         
Test Pano FDI Segmentation YOLOv8 - Malformed                         | PASS |
------------------------------------------------------------------------------
Test Production Detail                                                | PASS |
39 tests, 39 passed, 0 failed
==============================================================================
Output:  .\dentistry-inference-core\output.xml
Log:     .\dentistry-inference-core\log.html
Report:  .\dentistry-inference-core\report.html
```