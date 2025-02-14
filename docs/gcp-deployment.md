# GCP Deployment
## Build Docker Image
```bash
docker build -t gcr.io/sandbox-446907/dentistry-inference-core:latest
```
## Cloud Run
```bash
gcloud run deploy dentistry-inference-core --image gcr.io/sandbox-446907/dentistry-inference-core:latest --memory 32G
```
## API Gateway
```bash
gcloud api-gateway api-configs create dentistry-inference-core-api-config --api=dentistry-inference-core-api --openapi-spec=./conf/dentistry-inference-core-api.yaml --backend-auth-service-account=298229070754-compute@developer.gserviceaccount.com
```

## OpenAPI Spec
