"""Covid Radiology
- a
- b
- c
- d
"""

# streamlit configurations and options
import streamlit as st
from streamlit import caching
st.beta_set_page_config(page_title="Ex-stream-ly Cool App", page_icon="😎", layout="centered", initial_sidebar_state="expanded")
st.set_option('deprecation.showfileUploaderEncoding', False)

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from PIL import Image
import time
import random
import pandas as pd



#============================ About ==========================
def about():

    st.warning("""
    ## \u26C5 Behind The Scenes
        """)
    st.success("""
    To see how it works, please click the button below!
        """)
    github = st.button("👉🏼 Click Here To See How It Works")
    if github:
        github_link = "https://github.com/udaylunawat/Automatic-License-Plate-Recognition"
        try:
            webbrowser.open(github_link)
        except:
            st.error("""
                ⭕ Something Went Wrong!!! Please Try Again Later!!!
                """)
    st.info("Built with Streamlit by [Uday Lunawat 😎](https://github.com/udaylunawat)")


#================================= Functions =================================

def streamlit_preview_image(image):
    st.sidebar.image(
                image,
                use_column_width=True,
                caption = "Original Image")

def streamlit_output_image(image, caption):
    st.image(image,
            use_column_width=True,
            caption = caption)


#======================== Time To See The Magic ===========================

st.sidebar.markdown("## COVID-19 Classifier")
st.sidebar.markdown("Made with :heart: by [](https://udaylunawat.github.io)")

crop, image = None, None
img_size, crop_size = 600, 400

activities = ["Home", "Detector"]
choice = st.sidebar.radio("Go to", activities)

loaded_model = tf.keras.models.load_model("/content/output/models/base_model_covid.h5")

crop, image = None, None

st.write("## Upload your own image")

# placeholders
choose = st.empty() 
upload = st.empty()

predictor = st.checkbox("Make a Prediction 🔥")

samplefiles = sorted([sample for sample in listdir('data/sample_images')])
radio_list = ['Choose existing', 'Upload']

query_params = st.experimental_get_query_params()
# Query parameters are returned as a list to support multiselect.
# Get the second item (upload) in the list if the query parameter exists.
# Setting default page as Upload page, checkout the url too. The page state can be shared now!
default = 1

activity = choose.radio("Choose existing sample or try your own:", radio_list, index=default)

if activity:
    st.experimental_set_query_params(activity=radio_list.index(activity))
    if activity == 'Choose existing':
        selected_sample = upload.selectbox("Pick from existing samples", (samplefiles))
        image = Image.open('data/sample_images/'+selected_sample)
        IMAGE_PATH = 'data/sample_images/'+selected_sample
        image = Image.open('data/sample_images/'+selected_sample)
        img_file_buffer = None

    else:
        # You can specify more file types below if you want
        img_file_buffer = upload.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)

        IMAGE_PATH = img_file_buffer
        try:
            image = Image.open(IMAGE_PATH)
        except:
            pass

        selected_sample = None

if image:
    
    st.sidebar.markdown("## Preview Of Selected Image! 👀")
    streamlit_preview_image(image)