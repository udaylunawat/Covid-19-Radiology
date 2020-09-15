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
import tensorflow as tf

# basic visualization package
import matplotlib.pyplot as plt
# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#============================ About ==========================
def about():

    st.info("Built with Streamlit by [XYZ 😎](https://github.com)")

def plot_map(df, col, pal):
    df = df[df[col]>0]
    fig = px.choropleth(df, locations="Country/Region", locationmode='country names', 
                  color=col, hover_name="Country/Region", 
                  title=col, hover_data=[col], color_continuous_scale=pal)
    return fig
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
st.sidebar.markdown("Made with :heart: by [XYZ](https://www.github.com)")

crop, image = None, None
img_size, crop_size = 600, 400

activities = ["Data Visualization","Detector"]
choice = st.sidebar.radio("Go to", activities)

class_dict = {0:'COVID19',
              1:'NORMAL',
              2:'PNEUMONIA'}

crop, image = None, None

if choice == "Detector":
    loaded_model = tf.keras.models.load_model("output/models/base_model_covid.h5")
    st.write("## Upload your own image")

    # placeholders
    choose = st.empty() 
    upload = st.empty()

    predictor = st.checkbox("Make a Prediction 🔥")

    samplefiles = sorted([sample for sample in listdir('sample_images')])
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
            image = Image.open('sample_images/'+selected_sample)
            IMAGE_PATH = 'sample_images/'+selected_sample
            image = Image.open('sample_images/'+selected_sample)
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

        if predictor:
            test_image = cv2.resize(np.array(image), (224,224),interpolation=cv2.INTER_NEAREST)
            test_image = np.expand_dims(test_image,axis=0)
            probs = loaded_model.predict(test_image)
            pred_class = np.argmax(probs)

            pred_class = class_dict[pred_class]

            st.success('Prediction: '+pred_class)

elif choice == "Data Visualization":
    full_table = pd.read_csv('data/0_raw/covid_19_clean_complete.csv')
    # st.title("Full data")
    # st.write(full_table)
    
    country_wise = pd.read_csv('data/0_raw/country_wise_latest.csv')
    country_wise = country_wise.replace('', np.nan).fillna(0)
    st.title("Country wise data")
    st.write(country_wise)

    st.write(plot_map(country_wise, 'Confirmed', 'matter'))
    st.write(plot_map(country_wise, 'Deaths', 'matter'))
    st.write(plot_map(country_wise, 'Deaths / 100 Cases', 'matter'))

