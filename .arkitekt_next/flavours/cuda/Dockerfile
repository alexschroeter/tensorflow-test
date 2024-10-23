FROM tensorflow/tensorflow:latest-gpu

RUN pip install "arkitekt-next[all]>=0.8.10"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
