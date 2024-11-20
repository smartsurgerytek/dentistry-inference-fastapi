# Setup Guide

Choose your preferred setup method, ordered from basic to advanced.

## 1. Virtual Environment (Basic)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
## 2. Docker Build (Intermediate)
```bash
docker build -t dental-xray-api .
docker run -p 8000:8000 dental-xray-api
```
## 3. Docker Compose (Recommended)
```bash
docker-compose -f docker-compose.dev.yml up
```