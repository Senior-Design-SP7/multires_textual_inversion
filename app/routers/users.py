from fastapi import APIRouter, Body, Request, Response, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from datetime import datetime, timedelta
from typing import Union
from jose import JWTError, jwt

from fastapi.encoders import jsonable_encoder
from typing import List

from ..dependencies import User, UserCreate, Token, TokenData

SECRET_KEY = 'dd4de48aa8b65494b204d20f97d23bd3650fecd995a5a43299b6462d3b049e4a'
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

router = APIRouter(prefix="/user")

#fix so usernames cannot be repeated
#check if user already exists
@router.post("/create", response_description="Create a new user", status_code=status.HTTP_201_CREATED, response_model=User)
def create_user(request: Request, user: UserCreate = Body(...)):
    user = jsonable_encoder(user)
    user_entry = request.app.database["userInfo"].find_one( {"email": user['email']} )
    if user_entry is not None:
        raise HTTPException(
             status_code=400,
             detail="User already exists"
         )
    user['salt'] = bcrypt.gensalt()
    user['password'] = bcrypt.hashpw(user["password"].encode('utf-8'), user['salt'])
    new_user = request.app.database["userInfo"].insert_one(user)
    created_user = request.app.database["userInfo"].find_one( {"_id": new_user.inserted_id} )
    created_user.pop("password")
    created_user.pop("salt")
    return created_user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@router.post("/token", response_model = Token)
async def send_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user_entry = request.app.database["userInfo"].find_one( {"email": form_data.username} )
    if user_entry is None:
        raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Incorrect username or password",
             headers={"WWW-Authenticate": "Bearer"},
         )
    if bcrypt.hashpw(form_data.password.encode('utf-8'), user_entry["salt"]) != user_entry["password"]:
        raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Incorrect username or password",
             headers={"WWW-Authenticate": "Bearer"},
         )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user_entry['email']}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type":"bearer"}


async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user_entry = request.app.database["userInfo"].find_one( {"email": token_data.username} )
    if user_entry is None:
        raise credentials_exception
    return user_entry


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user