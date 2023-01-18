# backend/main.py

import uuid

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
from typing import List
import os

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

app = FastAPI()

@app.post("/addConcept/")
def create_upload_dir(name: str, files: List[UploadFile]):
    os.system('mkdir ' + name) #replace this with actual storage system for images
    dir = os.getcwd() + "/" + name
    os.chdir(dir)
    for f in files:
        file_location = dir + "/" + f.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(f.file.read())

    return dir

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)