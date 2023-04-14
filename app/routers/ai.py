from diffusers import StableDiffusionControlNetPipeline
from botocore.exceptions import ClientError
import logging
import boto3
from io import BytesIO
import train_dreambooth
import torch
from diffusers import ControlNetModel
from pipeline import DreamBoothMultiResPipeline
from fastapi import APIRouter,  UploadFile, responses
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import os
from typing import List
import base64
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector
import cv2

router = APIRouter(prefix="/ai")


MODEL_NAME = "runwayml/stable-diffusion-v1-5"
BUCKET_NAME = "multires-100"

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# NOTE: ADD AWS CREDENTIALS TO ENVIRONMENT VARIABLES
# CREDENTIALS INCLUDE: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

def upload_s3(local_path, bucketname, s3):
    for path, dirs, files in os.walk(local_path):
        for file in files:
            file_s3 = os.path.normpath(path + '/' + file)
            file_local = os.path.join(path, file)
            print("Upload:", file_local, "to target:", file_s3, end="")
            s3.upload_file(file_local, bucketname, file_s3)
            print(" ...Success")

    return


def download_s3_folder(bucket_name, s3_folder, s3, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def key_s3_size(bucket_name, key):
    # Get the size of an object in an S3 bucket
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=key)
    for obj in response.get('Contents', []):
        if obj['Key'] == key:
            return obj['Size']
    return None

# short helper to format cmd for training new concepts
def trainConcept(conceptDir, conceptName):
    # check if user has already trained this concept on s3 bucket
    # if (key_s3_size(BUCKET_NAME, userID) != None):
    #     return False
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
				--max_train_steps=175 ".format(MODEL_NAME, conceptDir, conceptName)

    train_dreambooth.main(train_dreambooth.parse_args(args.split()))
    # upload to s3 bucket
    # s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
    #                   aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # try:
    #     upload_s3(
    #         f"dreambooth_outputs/multires_100/{conceptName}", BUCKET_NAME, s3)
    # except ClientError as e:
    #     logging.error(e)
    #     return False
    # delete local files
    # os.system(f"rm -rf dreambooth_outputs/multires_100/{userID}")
    return True

# POST request that receives images + concept name as input.
# Returns successfully once new concept has been created


@router.post("/addConcept/")
def create_upload_dir(name: str, files: List[UploadFile], userID: str):
    # check if concept already exists in mongoDB
    # TODO Make it so that we have a dictionary of trained concepts for each user on mongoDB
    # save images to directory for reference
    # replace this with actual storage system for images
    os.system('mkdir ' + name)
    dir = os.getcwd() + "/" + name
    for f in files:
        file_location = dir + "/" + f.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(f.file.read())

    # call training cmd for giannis's model
    success = trainConcept(dir, name)

    # delete local files
    os.system('rm -rf ' + name)
    return f"returns {success}"

# POST request that receives an image based on a prompt that's given


@router.post("/promptModel/")
def model_prompt(name: str, prompt: str, guidance: int, inference_steps: int, resolution: int = 0):
    # download model from s3 bucket
    # s3 = boto3.resource('s3')
    file_name = f"dreambooth_outputs/multires_100/{name}"
    # download_s3_folder(BUCKET_NAME, file_name, s3)
    # check that file was downloaded successfully
    if (os.path.exists(file_name) == False):
        return responses.Response(content="Model not found", media_type="text/plain")
    pipe = DreamBoothMultiResPipeline.from_pretrained(
        f"{file_name}", use_auth_token=True)
    pipe = pipe.to("cuda")
    image = pipe(f"An image of <{name}({resolution})> " + prompt, num_inference_steps=inference_steps, guidance_scale=guidance)[0]
    image.save(f"{name}.png")

    #buffered = BytesIO()
    #image.save(buffered, format="PNG")
    #img_str = base64.b64encode(buffered.getvalue())
    # delete local files
    #os.system(f"rm -rf {file_name}")

    return FileResponse(f"{name}.png")

# POST request that generates conditioned image to use later
@router.post("/getCondition/")
def get_condition(image: UploadFile, condition: str):
    img = Image.open(image.file)
    if condition == "openpose":
        model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        output = model(img)
        output.save("output.png")
        return FileResponse("output.png")
    elif condition == "canny":
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        output = cv2.Canny(open_cv_image, 100, 200)
        cv2.imwrite("condition.png", output)
        return FileResponse("condition.png")
    elif condition == "mlsd":
        model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    elif condition == "hed":
        model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
    
    output = model(img)
    output.save("condition.png")
    return FileResponse("condition.png")

# POST request that also guides the model based on pose
# User passes in a pose image and the model will be guided to generate an image with that pose
@router.post("/promptModelPose/")
def model_prompt_pose(name: str, prompt: str, image: UploadFile, model_choice: str, guidance: int, inference_steps: int, resolution: int = 0):
    # load pretrained model for user and controlnet model
    controlnet = ControlNetModel.from_pretrained(
        f"lllyasviel/sd-controlnet-{model_choice}", torch_dtype=torch.float16) # model choices include openpose, scribble, canny, mlsd, and hed
    # download model from s3 bucket
    # s3 = boto3.resource('s3')
    file_name = f"dreambooth_outputs/multires_100/{name}"
    # download_s3_folder(BUCKET_NAME, file_name, s3)
    # check that file was downloaded successfully
    if (os.path.exists(file_name) == False):
        return responses.Response(content="Model not found", media_type="text/plain")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"{file_name}", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    prompt = f"An image of <{name}(0)> {prompt}"
    # load pose image
    pose = Image.open(image.file)
    output = pipe(prompt, pose, num_inference_steps=inference_steps, guidance_scale=guidance)
    loc = f"out_image.png"
    output.images[0].save(loc)
    return FileResponse(loc)

# Get request that returns a list of all concepts that have been trained for a user
# TODO Make it so that we have a dictionary of trained concepts for each user on mongoDB
@router.get("/getConcepts/")
def get_concepts():
    return "Chinaya needs to implement this"
