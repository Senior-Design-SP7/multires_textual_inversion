from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from ..dependencies import User, UserUpdate

router = APIRouter(prefix="/user")

#get: check if user exists
#post: create new user (sign up)
#put: update existing user or sign in
#delte: delete existing user

@router.post("/", response_description="Create a new user", status_code=status.HTTP_201_CREATED, response_model=User)
def create_book(request: Request, user: User = Body(...)):
    user = jsonable_encoder(user)
    new_user = request.app.database["userInfo"].insert_one(user)
    created_user = request.app.database["userInfo"].find_one( {"_id": new_user.inserted_id} )

    return created_user

@router.post("/", response_description="Create a new user", status_code=status.HTTP_201_CREATED, response_model=User)
def create_book(request: Request, user: User = Body(...)):
    user = jsonable_encoder(user)
    new_user = request.app.database["userInfo"].insert_one(user)
    created_user = request.app.database["userInfo"].find_one( {"_id": new_user.inserted_id} )

    return created_user