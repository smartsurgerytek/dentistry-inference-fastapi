# Deployment Guide

This guide explains how to deploy the dentistry-inference-fastapi service.

## Quick Start with Docker Compose
### Start the Service
Deploy the service with a single command:
```bash
docker-compose up -d
```

This will:
- Build the FastAPI container image
- Start the service on port 8000
- Run in detached mode
### Verify Deployment
Check if the service is running:
```bash
docker-compose ps
```
Access the API documentation at: http://localhost:8000/docs

### Stop the Service
```bash
docker-compose down
```

### Additional Commands
View logs:
```bash
dokcer-compose logs -f
```