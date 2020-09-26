from tensorflow.keras.applications import VGG16

def predict_label(image):
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
    test_image = VGG16.preprocess_input(image)
    probs = model.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    return pred_class, probs