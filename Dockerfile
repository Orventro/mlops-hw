FROM python:3.10-bookworm

RUN apt update && apt install gcc

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

COPY train.py infer.py load.sh ./
COPY petfinder/ ./petfinder
