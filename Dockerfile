FROM python:3.9

ARG HUGGINGFACE_TOKEN
RUN echo "Token at build: ${HUGGINGFACE_TOKEN}"
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv


COPY ./src /workspace/src
COPY ./conf /workspace/conf

RUN uv pip install --system huggingface_hub

RUN python src/allocation/service_layer/download.py

COPY requirements.txt /workspace/requirements.txt

RUN uv pip install --system -r /workspace/requirements.txt

#RUN python src/allocation/service_layer/download.py

EXPOSE 8080
#CMD ["fastapi", "run", "/workspace/src/allocation/entrypoints/fast_api.py", "--port", "8080"]
CMD ["uvicorn", "src.allocation.entrypoints.fast_api:app", "--host", "0.0.0.0", "--port", "8080"]
