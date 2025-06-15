FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install uvicorn fastapi numpy onnx onnxruntime pandas  

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]