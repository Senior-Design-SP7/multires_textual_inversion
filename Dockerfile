FROM nvidia/cuda:12.1.0-base-ubuntu20.04

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV PATH ="${PATH}:/usr/local/cuda/lib64"

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

ENV HF_TOKEN=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
ENV AWS_ACCESS_KEY_ID=AKIAXMKAHTYLFDYNIJNB
ENV AWS_SECRET_ACCESS_KEY=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


CMD ["uvicorn", "--host", "0.0.0.0","--port", "8000", "app.main:app"]
