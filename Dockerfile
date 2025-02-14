FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /workspace

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

COPY ./models /workspace/models
COPY ./src /workspace/src
COPY ./conf /workspace/conf

CMD ["fastapi", "run", "/workspace/src/allocation/entrypoints/fast_api.py", "--port", "8080"]