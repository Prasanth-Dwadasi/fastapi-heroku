from turtle import onclick
import pandas as pd
import streamlit as st
import requests
from PIL import Image
import zipfile
import os
import imageio
import io
import base64
import pickle
from geopy.geocoders import Nominatim
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
from datetime import date
import pandas as pd
import json

with open('./credentials/fastapi-nowcast-946843442a99.json') as source:
    info = json.load(source)

credentials = service_account.Credentials.from_service_account_info(info)
projectid = "fastapi-nowcast"
client = bigquery.Client(credentials= credentials,project=projectid)



def main():

    st.set_page_config(layout="wide")

    header = st.container()
    welcome = st.container()
    auth = st.container()

    # col1, col2 = st.columns([1, 3])

    # add_selectbox = st.sidebar.selectbox(
    #     "How would you like to be contacted?",
    #     ("Email", "Home phone", "Mobile phone")
    # )

    with header:
        st.title('Welcome to DSP weather forecasting')
        st.write("--------------------------------------------------------------------")
        image = 'forecast.jpg'
        st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write("--------------------------------------------------------------------")

    with auth:
        st.header('Please enter your details')
        user = st.text_input('Enter username',"")
        user_password = st.text_input('Enter password',type='password')
        user_creds = {'username': f"{user}", "password": f"{user_password}"}

        sess = st.session_state
        if not sess:
            sess.authenticated = False
        login_col ,logout_col = st.columns([0.2,1])

        print(user_creds)
        if login_col.button("Login") or sess.authenticated:
            sess.authenticated = True
            #print("Hii", sess.authenticated)
            token = requests.post('http://127.0.0.1:8000/token',data = user_creds)
            #print(token.status_code)
            if logout_col.button('Log out'):
                sess.authenticated = False
                token = None
                jwtoken = None
                st.markdown('Log out successfully')
                return None

            if token.status_code == 200:
                with welcome:
                    st.subheader('Predict weather forecast using the ultimate ML Model')
                    st.write("--------------------------------------------------------------------")
                    st.write("Enter the Latitude and Longitude for weather forecast")
                    lat = st.number_input("Enter Latitude",min_value=-180.0, max_value=180.0,value=0.0, step=1.,format="%.2f")
                    long = st.number_input("Enter Longitude",min_value=-180.0, max_value=180.0,value=0.0, step=1.,format="%.2f")

                    st.subheader('OR')

                    location = st.text_input('Enter the Location',"New York")
                    latest_check = st.checkbox('Get latest images')
                    #print(latest_check)
                    today = date.today()
                    query = """ SELECT COUNT(Username) FROM fastapi-nowcast.Logs.Usage_logs_record where Username = '{}' and Date = '{}'""".format(user,today)
                    query_job = client.query(query)
                    result = query_job.result()
                    no_requests_for_today = 0
                    for row in result:
                        print(row[0])
                        no_requests_for_today = int(row[0])
                    if(no_requests_for_today > 10):
                        st.markdown(f"MAXED OUT ON ATTEMPTS FOR THE TODAY")
                        st.button("Predict",disabled=True)
                    elif(0 <= no_requests_for_today <=10):

                        if st.button("Predict",disabled = False):
                            query1 = """ SELECT COUNT(Username) FROM fastapi-nowcast.Logs.Usage_logs_record where Username = '{}' and Date = '{}'""".format(user,today)
                            query1_job = client.query(query1)
                            result1 = query1_job.result()
                            for row in result1:
                                print(row[0])
                                no_requests_for_today = int(row[0])
                            if(no_requests_for_today > 10):
                                st.markdown(f"MAXED OUT ON ATTEMPTS FOR THE TODAY!!!")
                                st.button("Predict",disabled=True)
                            st.subheader('Predicted forecast')
                            # r = requests.post(f"http://127.0.0.1:8000/get_predictions/{location}")
                            if not location:
                                geolocator = Nominatim(user_agent="geoapiExercises")
                                location = geolocator.reverse(str(lat)+","+str(long))
                                st.write(f"Bsed on the latitude & longitude you entered the location is {location.address} ")
                                location = location.address.split(',')[0]
                            url = 'http://127.0.0.1:8000/get_predictions_json'
                            headers = {'location': location,'latest':latest_check}
                            r = requests.post(url, json = headers,stream=True)
                            today = date.today()
                            now = datetime.now()
                            query = """ INSERT INTO fastapi-nowcast.Logs.Usage_logs_record VALUES ('{}','{}','{}','{}')""".format(user,today,now,location)
                            query_job = client.query(query)
                            result = query_job.result()
                            print(result)
                            if 'Error' in str(r.content):
                                st.markdown("Incorrct location or location exceeds range")
                            else:
                                data_url = base64.b64encode(r.content).decode("utf-8")
                                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)

                            url2 = 'https://qdbhg7xfcl.execute-api.us-east-1.amazonaws.com/dev/qa'
                            x = []
                            with open('./narrative', 'rb') as fp:
                                x = (pickle.load(fp))
                            headers = {'episode_narrative': x[0]}
                            r2 = requests.post(url2, json = headers,stream=True)
                            st.write(r2.content)
                            headers = {'episode_narrative': x[1]}
                            r3 = requests.post(url2, json = headers,stream=True)
                            st.write(r3.content)

                            url3 = 'https://aw1rb3co58.execute-api.us-east-1.amazonaws.com/dev/ner'
                            headers = {'episode_narrative': x[0]}
                            r4 = requests.post(url3, json = headers,stream=True)
                            st.write(r4.content)

                            # string = str(r.content)
                            # if "Error" not in string:
                            #     with open("output.gif", 'wb') as f:
                            #         f.write(r.content)
                            # # im = Image.open(io.BytesIO(r.content))
                            # file_ = open("./output.gif", "rb")
                            # contents = file_.read()
                            #data_url = base64.b64encode(r.content).decode("utf-8")
                            # file_.close()
                            #st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)
            else:
                st.markdown(f"Login Failure ")







main()
                    # string = str(r.content)
                    # if "Error" not in string:
                    #     with open("output.zip", 'wb') as f:
                    #         f.write(r.content)
                    #     filepath = "./output"
                    #     with zipfile.ZipFile('output.zip', 'r') as zipObj:
                    #         zipObj.extractall(filepath)
                        
                    #     for file in os.listdir(filepath + "/images/"):
                    #         if file.endswith(".jpg"):
                    #             img = Image.open(filepath + "/images/" + file)
                    #             st.image(img)
                    # else:
                    #     st.write(""string"")

                
                


                    
                
            # def submit_add_project(project_name: str):
            #     """ Callback function during adding a new project. """
            #     # display a warning if the user entered an existing name
            #     if project_name in st.session_state.projects:
            #         st.warning(f'The name "{project_name}" is already exists.')
            #     else:
            #         st.session_state.projects.append(project_name)
            # new_project = st.text_input('New project name:',
            #                             key='input_new_project_name')
            # st.button('Add project', key='button_add_project',
            #           on_click=submit_add_project, args=(new_project, ))



'''
https://codelabs-preview.appspot.com/?file_id=1KH-V6PSW73jn9yeL-NzLM0T7C2pbLdTwGoSIT64zLmc#3
'''
