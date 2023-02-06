import uuid
from typing import Optional
from pydantic import BaseModel, Field

class User(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    firstName: str = Field(...)
    lastName: str = Field(...)
    email: str = Field(...)
    password: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "_id": "066de609-b04a-4b30-b46c-32537c7f1f6e",
                "firstName": "John",
                "lastName": "Doe",
                "email": "jd@meme.com",
                "password": "pwd"
            }
        }

class UserUpdate(BaseModel):
    firstName: str = Field(...)
    lastName: str = Field(...)
    email: str = Field(...)
    password: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "firstName": "John",
                "lastName": "Doe",
                "email": "jd@meme.com",
                "password": "pwd"
            }
        }