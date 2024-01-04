# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install . --no-cache-dir #(1)
RUN mkdir -p models/MyAwesomeModel/checkpoints
RUN mkdir -p reports/figures

ENTRYPOINT ["python", "-u", "src/train_model.py", "train", "--epochs", "3"]