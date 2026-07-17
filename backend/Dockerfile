FROM python:3.10-slim

WORKDIR /app

# System dependencies (for potential C++ extensions in torch/peft)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Need ML pipeline requirements as well for evaluation/inference imports
COPY ../ml_pipeline/requirements.txt /app/ml_pipeline_requirements.txt
RUN pip install --no-cache-dir -r /app/ml_pipeline_requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
