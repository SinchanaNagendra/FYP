# TFF comparison Docker container
# Uses a TF-compatible Python 3.10 image since TFF pins TF version strictly
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY fl_tff/requirements.txt /app/fl_tff/requirements.txt
RUN pip install --no-cache-dir -r fl_tff/requirements.txt

COPY fl/ /app/fl/
COPY data/vocab/ /app/data/vocab/
COPY fl_tff/run_tff.py /app/fl_tff/run_tff.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "fl_tff/run_tff.py"]
