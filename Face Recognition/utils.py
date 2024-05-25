from typing import Tuple, Union, AnyStr
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np

import streamlit as st

def get_center_rect(img: np.ndarray, 
                    rect_size: Tuple[int, int]) -> Tuple[Tuple, Tuple]:
    center = img.shape
    w, h = rect_size
    x = max(center[1]/2 - w/2, 0)
    y = max(center[0]/2 - h/2, 0)
    p1 = (int(x), int(y))
    p2 = (int(x+w), int(y+h))
    print(p1, p2)
    return p1, p2

def crop_img(img: np.ndarray,
             p1: Tuple, p2: Tuple) -> np.ndarray:
    return img[p1[1]:p2[1], p1[0]:p2[0]]
    

def preprocess_image(image: Union[np.ndarray, AnyStr], 
                     shape=(160,160)) -> np.ndarray:
    print(f"Type: {type(image)}")
    if isinstance(image, np.ndarray):
        print("Is numpy array")
        
        img = tf.keras.preprocessing.image.smart_resize(
            img, shape)
    # If image is path
    else:
        img = tf.keras.preprocessing.image.load_img(
            image, target_size=(160, 160))
        img = np.array(img)
    return img

def img_to_encoding(img: np.ndarray, 
                    model) -> np.ndarray:
    img = np.around(img / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)

    return embedding / np.linalg.norm(embedding, ord=2)



def who_is_it(image: Union[np.ndarray, AnyStr], database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.

    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    preprocessed_img = preprocess_image(
        image, shape=(160,160)
    )
    # st.image(preprocessed_img, caption="Antes modelo")
    encoding = img_to_encoding(preprocessed_img, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist =  np.linalg.norm(encoding - db_enc)

        print(f"Name: {name}, distance: {dist:.4f}")

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name
    ### END CODE HERE

    # if min_dist > 0.7:
    #     print("Not in the database.")
    # else:
    #     print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


def load_model(model_folder: Union[str, Path]):
    if not isinstance(model_folder, Path):
        model_folder = Path(model_folder)

    json_file = open(model_folder / 'model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_folder / 'model.h5')

    # Load the Keras model from the specified path
    return model