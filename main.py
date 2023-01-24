# backend/main.py

import uuid

import cv2
import uvicorn
from fastapi import FastAPI, UploadFile
import numpy as np
from PIL import Image
from typing import List
import os

MODEL_NAME="runwayml/stable-diffusion-v1-5"

#short helper to format cmd for training new concepts
def trainConcept(conceptDir, conceptName):
    trainCMD = "accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path={} \
                --instance_data_dir={} \
                --output_dir=dreambooth_outputs/multires_100/{} \
                --instance_prompt='S' \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=4 --gradient_checkpointing \
                --use_8bit_adam \
                --learning_rate=5e-6 \
                --lr_scheduler='constant' \
                --lr_warmup_steps=0 \
                --max_train_steps=100 ".format(MODEL_NAME, conceptDir, conceptName)
    return trainCMD

app = FastAPI()

# POST request that receives images + concept name as input.
# Returns successfully once new concept has been created
# TODO Make it so that we save the embeddings + a unique userid to our database
@app.post("/addConcept/")
def create_upload_dir(name: str, files: List[UploadFile]):
    # save images to directory for reference
    os.system('mkdir ' + name) #replace this with actual storage system for images
    dir = os.getcwd() + "/" + name
    for f in files:
        file_location = dir + "/" + f.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(f.file.read())

    # call training cmd for giannis's model
    # os.system(trainConcept(dir, name))
    return "SUCCESS!"

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)