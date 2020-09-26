import cv2
from PIL import Image
import numpy as np
from src.config import class_dict

def predict_label(image, model):
    '''
    Preprocesses the image for inference and predicts label and probabilities for each class.

    Parameters
    ----------
    image : numpy like image for prediction


    Returns
    -------
    pred_class : string like output for model prediction class 
    probs : array like output of model prediction probabilities
    '''
    # image = cv2.imread(file_path)
    test_image = image.convert('RGB') # get RGB PNG image
    test_image = cv2.resize(np.array(test_image), (224,224),interpolation=cv2.INTER_NEAREST)
    test_image = np.expand_dims(test_image,axis=0)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    return pred_class, probs