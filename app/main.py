# backend/main.py
import uvicorn
from fastapi import FastAPI
from pymongo import MongoClient

from .routers import users
from .routers import ai

app = FastAPI()


app.include_router(ai.router)
app.include_router(users.router)


@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient("mongodb://pnelakon:dimakisrocks@ac-qeu9obx-shard-00-00.z5fpuh8.mongodb.net:27017,ac-qeu9obx-shard-00-01.z5fpuh8.mongodb.net:27017,ac-qeu9obx-shard-00-02.z5fpuh8.mongodb.net:27017/?ssl=true&replicaSet=atlas-14pew5-shard-0&authSource=admin&retryWrites=true&w=majority")
    app.database = app.mongodb_client["users"]
    print("Connected to the MongoDB database!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()

@app.get("/")
def read_root():
	return {"message": "Welcome from the API"}

if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=8000)
