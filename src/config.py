import datetime

DATA_DIR = 'data/0_raw/COVID-19 Radiography Database'
PROCESSED_DATA_PATH = 'data/3_processed/data.csv'
PRETRAINED_MODEL = 'output/models/inference/base_model_covid.h5'


#============================ Live Data config ==========================
# https://rapidapi.com/astsiatsko/api/coronavirus-monitor
rapid_api_key = "dd8d4e05e8mshc5ab62dcd8a5f08p14b028jsna2726a63a74d"


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