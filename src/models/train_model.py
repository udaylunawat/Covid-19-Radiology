import pandas as pd
import joblib
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import backend as k
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout

from src.config import BATCH_SIZE, IMG_SIZE, LR, EPOCHS, LOG_DIR, CHECKPOINT_DIR, PROCESSED_DATA_PATH



def VGG16_model():
    base = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    output = base.layers[-1].output
    output = Flatten()(output)
    
    model = Model(base.input, outputs=output)
    
    for layer in model.layers:
        layer.trainable = False
    
    return model

data = pd.read_csv(PROCESSED_DATA_PATH)
data = shuffle(data)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
                             rotation_range=25,
                             fill_mode="nearest")

train_generator = datagen.flow_from_dataframe(
    dataframe = data,
    x_col = "path", y_col = "label",
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = True,
    subset = 'training',
    class_mode = 'categorical'
    )

validation_generator = datagen.flow_from_dataframe(
    dataframe = data,
    x_col = "path", y_col = "label",
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False,
    subset='validation',
    class_mode='categorical'
    )

TRAIN_SPE = train_generator.samples//BATCH_SIZE
VAL_SPE = validation_generator.samples//BATCH_SIZE

model = Sequential()
model.add(VGG16_model())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# model.summary()

k.clear_session()

print("Training Model")
optimizer = Adam(lr = LR, decay = LR/EPOCHS)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_accuracy', min_delta = 0.002, 
                          patience = 15, mode = 'auto', verbose = 1)

tensorboard_callback =TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

checkpointer = ModelCheckpoint(filepath = CHECKPOINT_DIR, monitor = 'val_accuracy', 
                               verbose = 0, save_best_only = True, mode ='auto')

callbacks_list = [earlystop, checkpointer, tensorboard_callback]

history = model.fit(train_generator, 
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=TRAIN_SPE,
                    validation_data=validation_generator,
                    validation_steps=VAL_SPE,
                    epochs=EPOCHS,
                    verbose=1, callbacks = callbacks_list)

print("Training Completed")
print("Saving Model as model.h5 in output/models directory")
model.save('output/models/model.h5')
  
# Save the model as a pickle in a file
print("Saving model metrics as output/history.pkl")
joblib.dump(history.history, 'output/history.pkl')