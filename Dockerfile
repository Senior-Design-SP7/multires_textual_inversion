FROM nvidia/cuda:12.1.0-base-ubuntu20.04

RUN apt update -y

RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.7 -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

EXPOSE 8000

WORKDIR /home

COPY ./requirements.txt .

COPY . .
RUN python3.7 --version
RUN apt install python3-pip -y
RUN apt install python3.7-distutils -y
RUN apt-get install python3.7-dev -y
RUN python3.7 -m pip install -r requirements.txt
CMD ["uvicorn", "--host", "0.0.0.0","--port", "8000", "app.main:app"]
