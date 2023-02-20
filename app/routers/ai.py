from fastapi import APIRouter,  UploadFile
import numpy as np
from PIL import Image
import os
from typing import List

router = APIRouter(prefix="/ai")

from pipeline import DreamBoothMultiResPipeline
import torch
import train_dreambooth

MODEL_NAME="runwayml/stable-diffusion-v1-5"

#short helper to format cmd for training new concepts
def trainConcept(conceptDir, conceptName):
	args = "--pretrained_model_name_or_path={} \
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

	train_dreambooth.main(train_dreambooth.parse_args(args.split()))
	return True


# POST request that receives images + concept name as input.
# Returns successfully once new concept has been created
# TODO Make it so that we save the embeddings + a unique userid to our database
@router.post("/addConcept/")
def create_upload_dir(name: str, files: List[UploadFile]):
	# check if concept already exists in mongoDB
	# TODO Make it so that we have a dictionary of trained concepts for each user on mongoDB

	# save images to directory for reference
	os.system('mkdir ' + name) #replace this with actual storage system for images
	dir = os.getcwd() + "/" + name
	for f in files:
		file_location = dir + "/" + f.filename
		with open(file_location, "wb+") as file_object:
			file_object.write(f.file.read())

	# call training cmd for giannis's model
	success = trainConcept(dir, name)
	return "SUCCESS!"

# POST request that receives an image based on a prompt that's given
# TODO Make it so that we have a dictionary of trained concepts for each user on mongoDB
# TODO Make the prompt customizable
@router.post("/promptModel/")
def model_prompt(name: str, location: str):
	pipe = DreamBoothMultiResPipeline.from_pretrained(f"{location}", use_auth_token=True)
	pipe = pipe.to("cuda")
	image = pipe(f"An image of <{name}(0)>")[0]
	loc = f"out_image.png"
	image.save(loc)
	return "SUCCESS!"
