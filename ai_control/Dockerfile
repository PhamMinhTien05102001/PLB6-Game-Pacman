FROM python:3.10.8-bullseye
WORKDIR /ai_control
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "server/server.py"]