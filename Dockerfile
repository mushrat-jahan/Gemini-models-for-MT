FROM python:3.12

WORKDIR /bangla-tranlator-backend

COPY . .

RUN pip install -r requirements.txt

CMD ['uvicorn', 'main:app', '--reload']
