import datetime

#============================ Paths ==========================

DATA_DIR = 'data/0_raw/COVID-19 Radiography Database'
PROCESSED_DATA_PATH = 'data/3_processed/data.csv'
PRETRAINED_MODEL = 'output/models/inference/base_model_covid.h5'

#============================ Live Data config ==========================

# https://rapidapi.com/astsiatsko/api/coronavirus-monitor
rapid_api_key = "dd8d4e05e8mshc5ab62dcd8a5f08p14b028jsna2726a63a74d"


#============================ Samples Links ==========================
sample_images_dict = {
    'covid_sample1':'https://i.imgur.com/86SEF4p.png',
    'covid_sample2':'https://i.imgur.com/E2G2vWm.png',
    'covid_sample3':'https://i.imgur.com/lrDcWOS.png',
    'covid_sample4':'https://i.imgur.com/87vLvst.png',
    'covid_sample5':'https://i.imgur.com/fj7rvy9.png',
    'normal_sample1':'https://i.imgur.com/JniKpBh.png',
    'normal_sample2':'https://i.imgur.com/tG0fzrZ.png',
    'normal_sample3':'https://i.imgur.com/UvGdCxp.png',
    'normal_sample4':'https://i.imgur.com/rcnjf5B.png',
    'normal_sample5':'https://i.imgur.com/b6Cjnk7.png',
    'pneumonia_sample1':'https://i.imgur.com/oH7ybcm.png',
    'pneumonia_sample2':'https://i.imgur.com/W3ize8y.png',
    'pneumonia_sample3':'https://i.imgur.com/KdZ7ko0.png',
    'pneumonia_sample4':'https://i.imgur.com/OFl077V.png',
    'pneumonia_sample5':'https://i.imgur.com/Vdv8VAm.png',
}
#============================ Model config ==========================
class_dict = {0:'COVID-19',
              1:'NORMAL',
              2:'Viral Pneumonia'}
              
BATCH_SIZE = 64
IMG_SIZE = 224
LR = 0.0001
EPOCHS = 20

LOG_DIR = "output/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_DIR = 'output/models/snapshots/model-{epoch:03d}-{val_accuracy:03f}.h5'