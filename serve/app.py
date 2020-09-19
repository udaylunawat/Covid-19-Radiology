"""Covid Radiology
- Project Description
- Covid Description
- DL Description
- VGG16 Description
"""

# streamlit configurations and options
import streamlit as st
from streamlit import caching
st.beta_set_page_config(page_title="Covid-19", page_icon="😎", layout="centered", initial_sidebar_state="expanded")
st.set_option('deprecation.showfileUploaderEncoding', False)

from os import listdir
from os.path import isfile, join
import cv2
import joblib
import time
import random
import numpy as np
from PIL import Image
import requests
import pandas as pd
from tensorflow.keras import backend as k
from tensorflow.keras.models import load_model

import io

import seaborn as sns
import matplotlib.pyplot as plt

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.visualization.visualize import metrics_plotly, plot_map, counts_bar
from src.data.make_dataset import live_data
from src.data.preprocess import covid_stats
from src.config import rapid_api_key, PRETRAINED_MODEL, PROCESSED_DATA_PATH, class_dict

#============================ About ==========================
def about():

    st.info("Built with Streamlit by [Uday 😎](http://udaylunawat.github.io/)")

#================================= Functions =================================
# @st.cache(suppress_st_warning=True)
def streamlit_preview_image(image):
    st.image(
                image,
                width =img_size,
                caption = "Image Preview")



#======================== Time To See The Magic ===========================

response = live_data(rapid_api_key)
data = pd.read_csv(PROCESSED_DATA_PATH)
# Load the history from the file 
history = joblib.load('output/history.pkl')
image = None, None
img_size = 400

st.sidebar.markdown("## COVID-19 Radiology")
st.sidebar.markdown("Made with :heart: by [Uday Lunawat](http://udaylunawat.github.io/)")


st.sidebar.info(__doc__)
activities = ["Data Visualization","Detector","Performance Metrics","About"]
choice = st.sidebar.radio("Go to", activities)

k.clear_session()
model = load_model(PRETRAINED_MODEL)
if choice == "Detector":

    st.write("## Upload your own image")

    # placeholders
    choose = st.empty() 
    upload = st.empty()

    predictor = st.button("Make a Prediction 🔥")

    sample_dir = 'data/sample_images/' 
    samplefiles = sorted([sample for sample in listdir(sample_dir)])
    upload_options = ['Choose existing', 'Upload','URL']

    query_params = st.experimental_get_query_params()
    # Query parameters are returned as a list to support multiselect.
    # Get the second item (upload) in the list if the query parameter exists.
    # Setting default page as Upload page, checkout the url too. The page state can be shared now!
    default = 0

    activity = choose.selectbox("Choose existing sample or try your own:", upload_options, index=default)

    if activity:
        st.experimental_set_query_params(activity=upload_options.index(activity))
        if activity == 'Choose existing':
            selected_sample = upload.selectbox("Pick from existing samples", (samplefiles))
            image = Image.open(sample_dir + selected_sample)
            IMAGE_PATH = sample_dir + selected_sample
            image = Image.open(sample_dir + selected_sample)
            img_file_buffer = None

        elif activity == 'Upload':
            image = None

            # You can specify more file types below if you want
            img_file_buffer = upload.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'], multiple_files = True)

            IMAGE_PATH = img_file_buffer
            try:
                image = Image.open(IMAGE_PATH)
            except:
                pass

            selected_sample = None

        elif activity == 'URL':
            image = None
            IMAGE_PATH = upload.text_input('URL to Image Address:')
            try:
                response = requests.get(IMAGE_PATH)
                image_bytes = io.BytesIO(response.content)
                image = Image.open(image_bytes)
            except:
                pass
            selected_sample, img_file_buffer = None, None

    if image:
        
        st.markdown("## Preview Of Selected Image! 👀")
        streamlit_preview_image(image)

        if predictor:
            test_image = cv2.resize(np.array(image), (224,224),interpolation=cv2.INTER_NEAREST)
            test_image = np.expand_dims(test_image,axis=0)
            probs = model.predict(test_image)
            pred_class = np.argmax(probs)
            pred_class = class_dict[pred_class]

            st.sidebar.success('Prediction: '+pred_class)


elif choice == "Data Visualization":

    country_wise, updated_at = covid_stats(response)
    st.title("Country wise data")
    st.write("**Data Updated at**: {}".format(updated_at))
    st.write(country_wise)

    st.title("Covid Live maps")
    map_option = st.selectbox("Select map type",['cases','deaths','deaths_per_1m_population'])
    st.write(plot_map(country_wise, map_option))


elif choice == "Performance Metrics":
    
    labels = list(data['label'].value_counts().keys())
    label_counts = data['label'].value_counts().values
    st.write(counts_bar(data, labels, label_counts))

    st.write(metrics_plotly(history,metrics = ['accuracy','loss','val_accuracy','val_loss'], title = 'Accuracy & Loss Plot'))
    st.write(metrics_plotly(history,metrics = ['accuracy','val_accuracy'], title = 'Accuracy Plot'))
    st.write(metrics_plotly(history,metrics = ['loss','val_loss'], title = 'Loss Plot'))
    st.image('output/figures/cm.png')
    st.sidebar.markdown("### Prediction Preview")
    st.sidebar.image('output/figures/pred.png', width = 300)

elif choice == "About":
    about()

