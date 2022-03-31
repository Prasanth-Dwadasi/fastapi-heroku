from datetime import datetime
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

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from display import get_cmap


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
print(event_id)
print(hf.keys())
hf.close()

id_available = [int((re.findall("[0-9]+", str(id)))[0])   for id in id_available]

catalog_mod = catalog.loc[catalog['event_id'].isin(id_available)]
catalog_mod['lat'] = catalog_mod.apply(lambda x : (x['llcrnrlat'] + x['urcrnrlat'])/2, axis=1)
catalog_mod['lon'] = catalog_mod.apply(lambda x : (x['llcrnrlon'] + x['urcrnrlon'])/2, axis=1)
catalog_mod['event_id'] = catalog_mod['event_id'].astype(int)


norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [ [82,82,82], [252,141,89],[255,255,191],[145,191,219]])/255


app = FastAPI()

@app.get("/")
def read_main():
    return {"message":"Pass the location to /get_predictions_json to get output"}

class Inputs(BaseModel):
    location: str
    starttime: Optional[datetime] = None

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
    file = predict(location)
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
  
def predict(location):
    # download_model()
    lat,lon = get_latlong(location)
    closest_distances = distanceCal(lat,lon)
    closest_distances = closest_distances[0:3]
    if closest_distances[0][1] >= 500:
        return None
    print(f"closest distances are {closest_distances}")
    nearest_loc_eventids = [x[0] for x in closest_distances]

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

predict('New York')
