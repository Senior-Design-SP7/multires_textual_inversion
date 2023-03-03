FROM python:3.7 
# Image from dockerhub

RUN apt update -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

EXPOSE 8000 
#Expose the port 8000 in which our application runs
WORKDIR / 
# Make /app as a working directory in the container
# Copy requirements from host, to docker container in /app 
COPY ./requirements.txt .
# Copy everything from ./src directory to /app in the container
COPY ./ . 
RUN python3 --version
RUN python3 -m pip install -r requirements.txt # Install the dependencies
# Run the application in the port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
