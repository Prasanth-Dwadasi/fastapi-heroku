from random import randrange
import pandas as pd
import urllib.request
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
from operator import itemgetter
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

download_model()

mse_file  = './mse_model.h5'
mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})

catalog = pd.read_csv("./CATALOG.csv")
files = list(catalog[catalog.event_id == 835047].file_name)

def download_h5():
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

# from nowcast_reader import read_data
# x_test,y_test = read_data('./nowcast_testing.h5',end=50)

# print(x_test.ndim)
# print(x_test.shape)

norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [ [82,82,82], [252,141,89],[255,255,191],[145,191,219]])/255


# models = [mse_model]

# for i,m in enumerate(models):
#         yp = m.predict(x_test)
#         if isinstance(yp,(list,)):
#             yp=yp[0]
#         y_preds.append(yp*norm['scale']+norm['shift'])

# locations = ['South Dakota','Nebraska','Kentucky','Vermont','Oregon']

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
    download_model()
    download_h5()
    lat,lon = get_latlong(location)
    closest_distances = distanceCal(lat,lon)
    closest_distances = closest_distances[0:3]
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

# def predict(location):
#     lat,lon = get_latlong(location)
#     closest_distances = distanceCal(lat,lon)
#     closest_distances = closest_distances[0:3]
#     print(f"closest distances are {closest_distances}")
#     nearest_loc_eventids = [x[0] for x in closest_distances]
#     loc_index = [id_available.index(ind) for evt in nearest_loc_eventids for ind in id_available if evt == ind]
#     print(f"loc index is {loc_index}")
#     x_test = getinput_images(loc_index[0])
#     x_test = np.asarray(x_test)
#     x_test = np.expand_dims(x_test, axis=0)
#     x_test = np.transpose(x_test, (0, 2, 3, 1))
#     # show_xtest(x_test)
#     print(f"x_test shape is {x_test.shape}")
#     yp = mse_model.predict(x_test)
#     y_preds= []
#     if isinstance(yp,(list,)):
#         yp=yp[0]
#     y_preds.append(yp*norm['scale']+norm['shift'])
    
#     file = save_images(y_preds)
#     return file







# def show_xtest(x_test):
#     print(f"x_test shape is {x_test.shape}")
#     x_test = x_test[0]
#     for i in range(13):
#         data_y = x_test[:,:,i]
#         plt.imshow(data_y, interpolation='nearest')
#         filepath = "./xtest_images/"
#         if not os.path.isdir(filepath):
#             os.mkdir(filepath)
#         plt.imsave(f"./xtest_images/{i}.jpg",data_y)






# def predict(location):
#     id = get_id(location)
#     if id == None:
#         return None
#     # if  y_preds.size==0:
#     yp = mse_model.predict(x_test)
#     if isinstance(yp,(list,)):
#         yp=yp[0]
#     y_preds.append(yp*norm['scale']+norm['shift'])
    
#     file = save_images(id,y_preds)
#     return file

# def save_images(id,y_preds):
#     y_preds = np.asarray(y_preds)
#     y_preds = y_preds[0]
#     for i in range(12):
#         data_y = y_preds[id,:,:,i]
#         filepath = "./images/"
#         if not os.path.isdir(filepath):
#             os.mkdir(filepath)
#         plt.imsave(f"./images/{i}.jpg",data_y)
#     return filepath


# for i in range(12):
#   data_y = y_preds[1,:,:,i]
#   plt.imshow(data_y, interpolation='nearest')
#   if not os.path.isdir("./images"):
#     os.mkdir("./images")
#   plt.imsave(f"./images/{i}.jpg",data_y)
#   plt.show()

