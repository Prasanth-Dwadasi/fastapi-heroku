from datetime import datetime, timedelta
from fastapi import FastAPI,File, UploadFile
from fastapi.responses import FileResponse,Response
# import os
# import zipfile
# import io
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
# from assignment3 import predict
from fastapi.responses import StreamingResponse


# from random import randrange
import pandas as pd
# import urllib.request
import os
import time
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from geopy.geocoders import Nominatim
import boto3
from botocore.handlers import disable_signing
import re
# from operator import itemgetter
from geopy import distance
import imageio

'''
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from display import get_cmap
'''

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import pickle

import warnings
warnings.filterwarnings('ignore')

def download_file(url):
    os.system(f'wget {url}')

def download_model():
    if not os.path.exists("./mse_model.h5"):
        download_file("https://www.dropbox.com/s/95vmmlci5x3acar/mse_model.h5?dl=0")

#download_model()

mse_file  = './mse_model.h5'
mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})

catalog = pd.read_csv("./CATALOG.csv")
files = list(catalog[catalog.event_id == 835047].file_name)

storm = pd.read_csv("./StormEvents_details-ftp_v1.0_d2019_c20220330.csv")


resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir')

for file in files:
    key = 'data/' + file
    filename = file.split('/')
    if 'VIL' in filename[2] and not os.path.exists("./SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"):
        bucket.download_file(key,filename[2])

files = [file.split('/')[2] for file in files]

id_available = []
hf = h5py.File('./SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5','r')
# with h5py.File(files[0],'r') as hf:
event_id = hf.get('id')
for i in range(851):
  id_available.append(event_id[i])
hf.close()

id_available = [int((re.findall("[0-9]+", str(id)))[0])   for id in id_available]

catalog_mod = catalog.loc[catalog['event_id'].isin(id_available)]
catalog_mod['lat'] = catalog_mod.apply(lambda x : (x['llcrnrlat'] + x['urcrnrlat'])/2, axis=1)
catalog_mod['lon'] = catalog_mod.apply(lambda x : (x['llcrnrlon'] + x['urcrnrlon'])/2, axis=1)
catalog_mod['event_id'] = catalog_mod['event_id'].astype(int)


norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [ [82,82,82], [252,141,89],[255,255,191],[145,191,219]])/255

#-------------------------------------------------------------------------------
# to get a string like this run:
# openssl rand -hex 16
SECRET_KEY = "beb63442a038034dde1ecf7df3ab1296"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


users_db = {
    "bigdata": {
        "username": "Bigdata",
        "full_name": "DSP",
        "email": "dspbigdata@gmail.com",
        "hashed_password": "$2b$12$a0ECCY16Y6VjSo5XE2uX0OLPExxwGoVcj4DCOy5I0PjjLh3U8.FIq",
        "disabled": False,
    },

    "Admin": {
        "username": "admin",
        "full_name": "DSP ADMIN",
        "email": "dsp@gmail.com",
        "hashed_password": "$2b$12$wPBjMVSbTXRi3SF0IL6bneZhb2L7e5GZh1N3SgIZDQzk15KnEpRa.",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(users_db, username: str, password: str):
    user = get_user(users_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

    
#-------------------------------------------------------------------------------
app = FastAPI()

@app.get("/")
def read_main():
    return {"message":"Pass the location to /get_predictions_json to get output"}

class Inputs(BaseModel):
    location: str
    starttime: Optional[datetime] = None
    latest: Optional[bool] = False

#From Outh2Fastapi``
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    print('user', user)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

# @app.post("/get_predictions_json/")
# def get_predictions_json(input:Inputs):
#     input_dict = input.dict()
#     location = input_dict["location"]
#     file = predict(location)
#     if file:
#         return zipfiles(file)
#     else:
#         return {"Error":"Location not available"}

@app.post("/get_predictions_json/")
def get_predictions_json(input:Inputs):
    input_dict = input.dict()
    location = input_dict["location"]
    latest = input_dict['latest']
    file = ""
    if latest:
        file = predict(location)
    else:
        with open ('./outfile', 'rb') as fp:
            itemlist = pickle.load(fp)
            cached_time = itemlist[0]
            current_time = time.time()
            if current_time - cached_time < 1800:
                file = search_cache(location)
                if not file:
                    file = predict(location)
            else:
                file = predict(location)

        # file = search_cache(location)
        # if not file:
        #     file = predict(location)

    if file:
        with open(file, 'rb') as f:
            img_raw = f.read()
        byte_io = BytesIO(img_raw)
        return StreamingResponse(byte_io, media_type='image/gif')
    else:
        return {"Error":"Location not available"}

# @app.post("/get_predictions/{location}")
# def get_predictions(location):
#     file = predict(location)
#     return zipfiles(file)


@app.get("/get_predictions/{location}")
def get_predictions(location):
    file = predict(location)
    if file:
        with open(file, 'rb') as f:
            img_raw = f.read()
        byte_io = BytesIO(img_raw)
        return StreamingResponse(byte_io, media_type='image/gif')
    else:
        return {"Error":"Location not available or out of range"}


# def zipfiles(file_path):
#     zip_filename = "archive.zip"

#     s = io.BytesIO()
#     zf = zipfile.ZipFile(s, "w")

#     # for fpath in filenames:
#     #     # Calculate path for file in zip
#     #     fdir, fname = os.path.split(fpath)

#     #     # Add file, at correct path
#     #     zf.write(fpath, fname)

#     for file in os.listdir(file_path):
#         if file.endswith(".jpg"):
#             zf.write(file_path +"/" + file)

#     # Must close zip for all contents to be written
#     zf.close()

#     # Grab ZIP file from in-memory, make response with correct MIME-type
#     resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
#         'Content-Disposition': f'attachment;filename={zip_filename}'
#     })

#     return resp




# @app.post("/upload")
# async def upload(files: List[UploadFile] = File(...)):

#     # in case you need the files saved, once they are uploaded
#     for file in files:
#         contents = await file.read()
#         save_file(file.filename, contents)

#     return {"Uploaded Filenames": [file.filename for file in files]}

# def save_file(filename, data):
#     with open(filename, 'wb') as f:
#         f.write(data)


# @app.get("/download_images")
# def image_endpoint():
#     images = ["pic.jpg","Picture1.png"]
#     return zipfiles(images)

def distanceCal(lat,long):
    distances = {}
    given = (lat,long)
    for lat1,long1,eventid in zip(catalog_mod['lat'],catalog_mod['lon'],catalog_mod['event_id']):
        distances[eventid] = int(distance.distance(given, (lat1,long1)).miles)
    distances_sorted = (sorted(distances.items(), key=lambda item: item[1]))
    return (distances_sorted)

def distanceCal_cached(lat,long):
    distances = {}
    given = (lat,long)
    itemlist = []
    with open ('./outfile', 'rb') as fp:
        itemlist = pickle.load(fp)
    for i,latlongs in enumerate(itemlist[1:]):
        distances[i] = int(distance.distance(given, latlongs).miles)
    distances_sorted = (sorted(distances.items(), key=lambda item: item[1]))
    return (distances_sorted)


def get_latlong(adress):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(adress)  
    print((location.latitude, location.longitude))
    return location.latitude, location.longitude

def getinput_images(index):
    x_test = []
    with h5py.File(files[0],'r') as hf:
        event_id = hf['id'][index]
        vil = hf['vil'][index]
        for j in range(13):
            x_test.append(vil[:,:,j])
    return x_test

def get_location(lat,lon):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(str(lat)+","+str(lon))
    return location.address.split(',')[0]

def get_narratives(eventid):
    event_narrative = str(storm.EVENT_NARRATIVE[storm.EVENT_ID==eventid].iloc[0])
    episode_narrative = str(storm.EPISODE_NARRATIVE[storm.EVENT_ID==eventid].iloc[0])
    narratives = [event_narrative,episode_narrative]
    with open('./narrative', 'wb') as fp:
        pickle.dump(narratives, fp)

def predict(location):
    # download_model()
    lat,lon = get_latlong(location)
    closest_distances = distanceCal(lat,lon)
    closest_distances = closest_distances[0:3]
    if closest_distances[0][1] >= 500:
        return False
    print(f"closest distances are {closest_distances}")
    nearest_loc_eventids = [x[0] for x in closest_distances]
    get_narratives(nearest_loc_eventids[0])

    loc_index = [id_available.index(ind) for evt in nearest_loc_eventids for ind in id_available if evt == ind]
    print(f"loc index is {loc_index}")
    x_test = getinput_images(loc_index[0])
    x_test = np.asarray(x_test)
    x_test = np.expand_dims(x_test, axis=0)
    x_test = np.transpose(x_test, (0, 2, 3, 1))
    # show_xtest(x_test)
    print(f"x_test shape is {x_test.shape}")
    yp = mse_model.predict(x_test)
    y_preds= []
    if isinstance(yp,(list,)):
        yp=yp[0]
    y_preds.append(yp*norm['scale']+norm['shift'])
    y_preds = np.asarray(y_preds)
    y_preds = np.squeeze(y_preds, axis=1)
    print(y_preds.shape)
    y_preds = y_preds.reshape(384, 384, 12)
    filepath_gif = './ypred.gif'
    with imageio.get_writer(filepath_gif, mode='I') as writer:
        for i in range(12):
            # data_y = y_preds[:,:,i]
            writer.append_data(y_preds[:,:,i])
    # file = save_images(y_preds)
    print("Images came from model")
    return filepath_gif


def save_images(y_preds):
    y_preds = np.asarray(y_preds)
    y_preds = np.squeeze(y_preds, axis=1)
    print(y_preds.shape)
    y_preds = y_preds.reshape(384, 384, 12)
    for i in range(12):
        data_y = y_preds[:,:,i]
        filepath = "./images/"
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        plt.imsave(f"./images/{i}.jpg",data_y)
    return filepath

def search_cache(location):
    lat,lon = get_latlong(location)
    closest_distances = distanceCal_cached(lat,lon)
    if closest_distances[0][1] >= 200:
        return False
    else:
        filename = './output/' + 'ypred' + str(closest_distances[0][0]) + '.gif'
        print("Images came from cache",filename)
        return filename


#search_cache('California')

print("finished")




#predict('New York')
