FROM python:3.10.8-bullseye
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "server/server.py"]