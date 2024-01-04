# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/raw/random_mnist_images/ data/tst/
COPY models/MyAwesomeModel/checkpoints/ep3.pth models/

WORKDIR /
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "src/predict_model.py", "predict", "models/ep3.pth", "data/tst/mnist_sample.npy"]