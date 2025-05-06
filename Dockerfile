FROM python:3.9

ARG HUGGINGFACE_TOKEN
RUN echo "Token at build: ${HUGGINGFACE_TOKEN}"
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /workspace

COPY ./src /workspace/src
COPY ./conf /workspace/conf

RUN pip install huggingface_hub

RUN python src/allocation/service_layer/download.py

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

#RUN python src/allocation/service_layer/download.py


#CMD ["fastapi", "run", "/workspace/src/allocation/entrypoints/fast_api.py", "--port", "8080"]
CMD ["uvicorn", "src.allocation.entrypoints.fast_api:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080