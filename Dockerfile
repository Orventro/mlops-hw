FROM python:3.10-bookworm

RUN apt install gcc

COPY train.py infer.py requirements.txt load.sh ./
RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

COPY petfinder/ ./petfinder