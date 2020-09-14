from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from keras.utils import np_utils

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(y)
y_encoded = np_utils.to_categorical(encoded_labels)


X_train, X_cvtest, y_train, y_cvtest = train_test_split(X, y_encoded, test_size=0.40, stratify=y_encoded, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_cvtest, y_cvtest, test_size=0.50, stratify=y_cvtest, random_state=42)

del X
del y_encoded
del encoded_labels


BATCH_SIZE = 64
TRAIN_LENGTH = len(X_train)
VAL_LENGTH = len(X_cv)
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
SPE_VAL = VAL_LENGTH // BATCH_SIZE
IMG_SIZE = 224
LR = 0.0001
EPOCHS = 50

datagen = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

def VGG16_model():
    base = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    output = base.layers[-1].output
    output = Flatten()(output)
    
    model = Model(base.input, outputs=output)
    
    for layer in model.layers:
        layer.trainable = False
    
    return model

model = Sequential()
model.add(VGG16_model())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.summary()

tf.keras.backend.clear_session()

print("Training Model")
optimizer = Adam(lr = LR, decay = LR/EPOCHS)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_accuracy', min_delta = 0.002, patience = 5 ,mode = 'auto', verbose = 1)

ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
checkpointer = ModelCheckpoint(filepath = '/content/model-{epoch:03d}.h5', 
                               monitor = 'val_accuracy', verbose = 1, save_best_only = False, mode ='auto')

callbacks_list = [earlystop, checkpointer]

history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_data=(X_cv, y_cv),
                                    validation_steps=SPE_VAL,
                                    epochs=EPOCHS,
                                    verbose=1, callbacks = callbacks_list)

print("Training Completed")
print("Saving Model as base_model_covid.h5 in /output/models directory")
model.save('/content/output/models/base_model_covid.h5')