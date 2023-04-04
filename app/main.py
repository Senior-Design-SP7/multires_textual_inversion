# backend/main.py
import uvicorn
from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware

from .routers import users
from .routers import ai

from huggingface_hub.hf_api import HfFolder
import os

app = FastAPI()

app.include_router(ai.router)
app.include_router(users.router)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient("mongodb://pnelakon:dimakisrocks@ac-qeu9obx-shard-00-00.z5fpuh8.mongodb.net:27017,ac-qeu9obx-shard-00-01.z5fpuh8.mongodb.net:27017,ac-qeu9obx-shard-00-02.z5fpuh8.mongodb.net:27017/?ssl=true&replicaSet=atlas-14pew5-shard-0&authSource=admin&retryWrites=true&w=majority")
    app.database = app.mongodb_client["users"]
    print("Connected to the MongoDB database!")

@app.on_event("startup")
def startup_huggingface_client():
    # read environment variables for huggingface credentials
    token = os.environ.get("HF_TOKEN")
    HfFolder.save_token(token)
    print("Connected to the HuggingFace API!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()

@app.get("/")
def read_root():
	return {"message": "Welcome from the API"}

if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=8000)

# to run the app, run the following command:
# uvicorn app.main:app --reload