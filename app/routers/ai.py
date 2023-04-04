from fastapi import APIRouter,  UploadFile, responses
import numpy as np
from PIL import Image
import os
from typing import List

router = APIRouter(prefix="/ai")

from pipeline import DreamBoothMultiResPipeline
from diffusers import StableDiffusionControlNetPipeline 
from diffusers import ControlNetModel
import torch
import train_dreambooth
import boto3
import logging
from botocore.exceptions import ClientError

MODEL_NAME="runwayml/stable-diffusion-v1-5"
BUCKET_NAME = "multires"

# NOTE: ADD AWS CREDENTIALS TO ENVIRONMENT VARIABLES
# CREDENTIALS INCLUDE: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

def key_s3_size(bucket_name, key):
    # Get the size of an object in an S3 bucket
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=key)
    for obj in response.get('Contents', []):
        if obj['Key'] == key:
            return obj['Size']
    return None

#short helper to format cmd for training new concepts
def trainConcept(conceptDir, conceptName):
	# check if user has already trained this concept on s3 bucket
	if (key_s3_size(BUCKET_NAME, conceptName) != None):
		return False
	args = "--pretrained_model_name_or_path={} \
				--instance_data_dir={} \
				--output_dir=dreambooth_outputs/multires_100/{} \
				--instance_prompt='S' \
				--resolution=512 \
				--train_batch_size=1 \
				--gradient_accumulation_steps=4 --gradient_checkpointing \
				--use_8bit_adam \
				--learning_rate=5e-6 \
				--lr_scheduler=constant \
				--lr_warmup_steps=0 \
				--max_train_steps=100 ".format(MODEL_NAME, conceptDir, conceptName)

	train_dreambooth.main(train_dreambooth.parse_args(args.split()))
	# upload to s3 bucket
	s3 = boto3.resource('s3')
	try:
		response = s3.upload_file(f"dreambooth_outputs/multires_100/{conceptName}", BUCKET_NAME, conceptName)
	except ClientError as e:
		logging.error(e)
		return False
	# delete local files
	os.system(f"rm -rf dreambooth_outputs/multires_100/{conceptName}")
	return True

# POST request that receives images + concept name as input.
# Returns successfully once new concept has been created
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
	return f"returns {success}"

# POST request that receives an image based on a prompt that's given
@router.post("/promptModel/")
def model_prompt(name: str, prompt: str):
	# download model from s3 bucket
	s3 = boto3.resource('s3')
	file_name = f"dreambooth_outputs/multires_100/{name}"
	s3.download_file(BUCKET_NAME, name, file_name) 
	# check that file was downloaded successfully
	if (os.path.isfile(file_name) == False):
		return responses.Response(content="Model not found", media_type="text/plain")
	pipe = DreamBoothMultiResPipeline.from_pretrained(f"{file_name}", use_auth_token=True)
	pipe = pipe.to("cuda")
	image = pipe(f"An image of <{name}(0)>" + prompt)[0]
	# delete local files
	os.system(f"rm -rf {file_name}")
	# Return FileResponse
	return responses.Response(content=image.tobytes(), media_type="image/png")

# POST request that also guides the model based on pose 
# User passes in a pose image and the model will be guided to generate an image with that pose
@router.post("/promptModelPose/")
def model_prompt_pose(name: str, prompt: str, image: UploadFile):
	# load pretrained model for user and controlnet model
	# TODO make it so that choice of controlnet is customizable for now we're using lllyasviel/sd-controlnet-openpose
	controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
	# download model from s3 bucket
	s3 = boto3.resource('s3')
	file_name = f"dreambooth_outputs/multires_100/{name}"
	s3.download_file(BUCKET_NAME, name, file_name) 
	# check that file was downloaded successfully
	if (os.path.isfile(file_name) == False):
		return responses.Response(content="Model not found", media_type="text/plain")
	pipe = StableDiffusionControlNetPipeline.from_pretrained(
		f"{file_name}", controlnet=controlnet, torch_dtype=torch.float16
	)
	pipe = pipe.to("cuda")
	prompt = f"An image of <{name}(0)> {prompt}"
	# load pose image
	pose = Image.open(image.file)
	output = pipe(prompt, pose)
	loc = f"out_image.png"
	output.images[0].save(loc)
	return responses.Response(content=output.images[0].tobytes(), media_type="image/png")

# Get request that returns a list of all concepts that have been trained for a user
# TODO Make it so that we have a dictionary of trained concepts for each user on mongoDB
@router.get("/getConcepts/")
def get_concepts():
	return "Chinaya needs to implement this"
