from fastapi import APIRouter, Body, Request, Response, HTTPException, status, Depends
from fastapi.responses import RedirectResponse
import time
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from datetime import datetime, timedelta
from typing import Union
from jose import JWTError, jwt

from email_validator import validate_email, EmailNotValidError


from fastapi.encoders import jsonable_encoder
from typing import List

from ..dependencies import User, UserCreate, Token, TokenData

import stripe
stripe.api_key = 'sk_test_51MhgqVATxDijcQ83J0xDkP1Lx1KnDIQZN26HTcN20hNbGKYbC8S6GaIzZaNxqnI8oCPuASITqsfk5SiIezLyz6hB00Lel66x2h'


#add is paid field to database
#when signing up
#   No token is sent
#   User is initialized in backend with isPaid field=0 and the checkout session set to checkout session id
#   success page will be the login page
#   cancel page tbd

#when logging in
#   check if isPaid field is set to 1, if so login nomrally, send token
#   else if isPaid is set to 0, get checkout session id and check if it has been paid
#       if checkout session id shows it has been paid, update isPaid field to 1 and send token
#       else if checkout session id shows it has not been paid
#           expire old checkout session
#           generate new checkout session, replace database checkout session id with newly generated one. 
#           return redirect


SECRET_KEY = 'dd4de48aa8b65494b204d20f97d23bd3650fecd995a5a43299b6462d3b049e4a'
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/token")

router = APIRouter(prefix="/user")

#fix so usernames cannot be repeated
#check if user already exists
@router.post("/create", response_description="Create a new user")
def create_user(request: Request, user: UserCreate = Body(...)):
    user = jsonable_encoder(user)
    user_entry = request.app.database["userInfo"].find_one( {"email": user['email']} )
    if user_entry is not None:
        raise HTTPException(
             status_code=400,
             detail="User already exists"
        )

    try:
        # validate and get info
        v = validate_email(user['email'])
        # replace with normalized form
        user['email'] = v["email"] 
    except EmailNotValidError as e:
        # email is not valid, exception message is human-readable
        raise HTTPException(
             status_code=400,
             detail=str(e)
        )
        
    user['salt'] = bcrypt.gensalt()
    user['password'] = bcrypt.hashpw(user["password"].encode('utf-8'), user['salt'])
    user['isPaid'] = 0

    try:
        print("before checjout session")

        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': 'price_1MsxnyATxDijcQ83d7SMPz11',
                    'quantity': 1
                }
            ],
            mode='payment',
            success_url= 'https://google.com',  #change to success page that is made
            customer_email = user['email'],
            after_expiration={
                'recovery': {
                'enabled': True,
                },
            }
        )
        print("after checjout session")
        user["checkout"] = checkout_session.id
        new_user = request.app.database["userInfo"].insert_one(user)
        created_user = request.app.database["userInfo"].find_one( {"_id": new_user.inserted_id} )
        created_user.pop("password")
        created_user.pop("salt")
        print(checkout_session)
        return {"checkout_url": checkout_session.url}

    except Exception as e:
        return str(e)




def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


#when logging in
#   check if isPaid field is set to 1, if so login nomrally, send token
#   else if isPaid is set to 0, get checkout session id and check if it has been paid
#       if checkout session id shows it has been paid, update isPaid field to 1 and send token
#       else if checkout session id shows it has not been paid
#           expire old checkout session
#           generate new checkout session, replace database checkout session id with newly generated one. 
#           return redirect

@router.post("/token")
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
    
    if user_entry['isPaid'] == 1:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_entry['email']}, expires_delta=access_token_expires)
        return {"access_token": access_token, "token_type":"bearer"}

    elif user_entry['isPaid'] == 0:
        checkout_session = stripe.checkout.Session.retrieve(user_entry["checkout"])

        if checkout_session.payment_status == "unpaid":
            if checkout_session.status == "open":
                stripe.checkout.Session.expire(user_entry["checkout"])
            try:
                checkout_session = stripe.checkout.Session.create(
                    line_items=[
                        {
                            'price': 'price_1MsxnyATxDijcQ83d7SMPz11',
                            'quantity': 1,
                        },
                    ],
                    mode='payment',
                    success_url= 'https://google.com',  #change to success page that is made
                    customer_email = user_entry['email'],
                    after_expiration={
                        'recovery': {
                        'enabled': True,
                        },
                    }
                )
                #set the user['checkout'] to the new checkout id
                myquery = { "checkout": user_entry['checkout'] }
                newvalues = { "$set": { "checkout": checkout_session.id } }
                request.app.database["userInfo"].update_one(myquery, newvalues)
                return {"checkout_url": checkout_session.url}

            except Exception as e:
                return str(e)

        else:
            myquery = { "email": user_entry['email'] }
            newvalues = { "$set": { "isPaid": 1 } }
            request.app.database["userInfo"].update_one(myquery, newvalues)
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(data={"sub": user_entry['email']}, expires_delta=access_token_expires)
            return {"access_token": access_token, "token_type":"bearer"}       
            


#       if checkout session id shows it has been paid, update isPaid field to 1 and send token
#       else if checkout session id shows it has not been paid
#           expire old checkout session
#           generate new checkout session, replace database checkout session id with newly generated one. 
#           return redirect        


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

# return user concepts
@router.get("/concepts", response_model=List[str])
async def read_users_concepts(current_user: User = Depends(get_current_user)):
    return current_user['concepts']

# add a concept to user
@router.post("/concepts", response_model=List[str])
async def add_user_concept(request:Request ,current_user: User = Depends(get_current_user), concept: str = Body(...)):
    request.app.database["userInfo"].update_one( {"email": current_user['email']}, {"$push": {"concepts": concept}} )
    return request.app.database["userInfo"].find_one( {"email": current_user['email']} )['concepts']
