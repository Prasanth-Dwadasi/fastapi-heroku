from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pendulum
import os
import imageio
import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
import re
import random
import time
import pickle


import warnings
warnings.filterwarnings('ignore')

id_available = []
hf = h5py.File('/home/prasanthdwadasi/fastapi/fastapi-heroku/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5','r')
# with h5py.File(files[0],'r') as hf:
event_id = hf.get('id')
for i in range(851):
  id_available.append(event_id[i])
hf.close()
id_available = [int((re.findall("[0-9]+", str(id)))[0])   for id in id_available]

catalog = pd.read_csv("/home/prasanthdwadasi/fastapi/fastapi-heroku/CATALOG.csv")
catalog_mod = catalog.loc[catalog['event_id'].isin(id_available)]
catalog_mod['lat'] = catalog_mod.apply(lambda x : (x['llcrnrlat'] + x['urcrnrlat'])/2, axis=1)
catalog_mod['lon'] = catalog_mod.apply(lambda x : (x['llcrnrlon'] + x['urcrnrlon'])/2, axis=1)
catalog_mod['event_id'] = catalog_mod['event_id'].astype(int)

def getinput_images(index):
    x_test = []
    with h5py.File('/home/prasanthdwadasi/fastapi/fastapi-heroku/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5','r') as hf:
        event_id = hf['id'][index]
        vil = hf['vil'][index]
        for j in range(13):
            x_test.append(vil[:,:,j])
    return x_test

mse_file  = '/home/prasanthdwadasi/fastapi/fastapi-heroku/mse_model.h5'
mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})

print("mse model loaded")
norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [ [82,82,82], [252,141,89],[255,255,191],[145,191,219]])/255

latlong = []
def generate_cached_images():
    h5id = random.sample(range(0, 850), 50)
    for i,id in enumerate(h5id):
        eventid = id_available[id]
        temp = catalog_mod.loc[catalog_mod.event_id == eventid]
        lat,lon = temp.iloc[0,-2],temp.iloc[0,-1]
        latlong.append((lat,lon))
        if i==49 :
            latlong.insert(0, time.time())
            with open('/home/prasanthdwadasi/fastapi/fastapi-heroku/outfile', 'wb') as fp:
                pickle.dump(latlong, fp)
            print(latlong)
        x_test = getinput_images(id)
        x_test = np.asarray(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        x_test = np.transpose(x_test, (0, 2, 3, 1))

        #print((x_test.shape),i)

        x_test = x_test.reshape(384, 384, 13)
        #cwd = os.getcwd()
        cwd = '/home/prasanthdwadasi/fastapi/fastapi-heroku'
        os.makedirs(cwd + '/output', exist_ok=True)
        filepath_gif = cwd + '/output/ypred' + str(i) + '.gif'
        with imageio.get_writer(filepath_gif, mode='I') as writer:
            for i in range(12):
                writer.append_data(x_test[:,:,i])
        print(cwd)


with DAG(dag_id="generate_predictions_dag",
         start_date=pendulum.datetime(2022,3,31,tz='EST'),
         schedule_interval="@hourly",
         catchup=False) as dag:

        task1 = PythonOperator(
        task_id="generate_cached_images",
        python_callable=generate_cached_images)

task1
