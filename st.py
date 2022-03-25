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

st.set_page_config(layout="wide")

header = st.container()
welcome = st.container()
displayop = st.container()

# col1, col2 = st.columns([1, 3])
import streamlit as st

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

with header:
    st.title('Welcome to DSP weather forecasting')
    st.write("--------------------------------------------------------------------")
    image = 'forecast.jpg'
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # st.write("--------------------------------------------------------------------")


with welcome:
    st.subheader('Predict weather forecast using the ultimate ML Model')
    st.write("--------------------------------------------------------------------")
    st.write("Enter the Latitude and Longitude for weather forecast")
    lat = st.number_input("Enter Latitude",min_value=-180.0, max_value=180.0,value=0.0, step=1.,format="%.2f")
    long = st.number_input("Enter Longitude",min_value=-180.0, max_value=180.0,value=0.0, step=1.,format="%.2f")

    st.subheader('OR')

    location = st.text_input('Enter the Location',"New York")
    if st.button("Predict"):
        st.subheader('Predicted forecast')
        # r = requests.post(f"http://127.0.0.1:8000/get_predictions/{location}")
        url = 'http://127.0.0.1:8000/get_predictions_json'
        headers = {'location': location}
        r = requests.post(url, json = headers,stream=True)
        # string = str(r.content)
        # if "Error" not in string:
        #     with open("output.gif", 'wb') as f:
        #         f.write(r.content)
        # # im = Image.open(io.BytesIO(r.content))
        # file_ = open("./output.gif", "rb")
        # contents = file_.read()
        data_url = base64.b64encode(r.content).decode("utf-8")
        # file_.close()
        st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)













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