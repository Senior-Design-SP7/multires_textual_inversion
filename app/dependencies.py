import uuid
from typing import Optional
from pydantic import BaseModel, Field
from typing import Union


class User(BaseModel):
    firstName: str = Field(...)
    lastName: str = Field(...)
    email: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "firstName": "John",
                "lastName": "Doe",
                "email": "jd@meme.com",
            }
        }


class UserCreate(BaseModel):
    firstName: str = Field(...)
    lastName: str = Field(...)
    email: str = Field(...)
    password: str = Field(...)
    concepts: list = Field(...)
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "firstName": "John",
                "lastName": "Doe",
                "email": "jd@meme.com",
                "password": "pwd",
                "concepts": "[]"
            }
        }



class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Union[str, None] = None



class UserUpdate(BaseModel):
    firstName: str = Field(...)
    lastName: str = Field(...)
    email: str = Field(...)
    password: str = Field(...)
    concepts: list = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "firstName": "John",
                "lastName": "Doe",
                "email": "jd@meme.com",
                "password": "pwd",
                "concepts": "[]"
            }
        }