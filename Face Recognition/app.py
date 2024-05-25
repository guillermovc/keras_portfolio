from typing import Union, AnyStr, Tuple
from PIL import Image
import json
from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

from utils import *

@st.cache
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


# A simple CNN model with cached loading
class FaceModel:
    def __init__(self, 
                 model_folder="./keras-facenet-h5",
                 database_path="database.json"):
        # Load the cached model
        self.model = load_model(model_folder)
        
        # Create database
        with open(database_path, 'r') as file:
            self.database = json.load(file)

    def predict(self, image):
        min_dist, identity = who_is_it(
            image=image,
            database=self.database,
            model=self.model
        )

        return min_dist, identity

def predict_identity(captured_image):
    img = Image.open(captured_image)
    captured_image = np.array(img)
    p1, p2 = get_center_rect(captured_image, (400, 400))
    img_crop = crop_img(captured_image, p1, p2)

    # captured_image_zone = cv2.rectangle(captured_image.copy(), p1, p2, (255,0,0), thickness=3)
    # st.image(captured_image_zone, caption="Captured Image (red zone will be cropped)", use_column_width=True)

    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            plt.imsave("image.jpg", img_crop)
            # plt.imsave("image.jpg", captured_image)
            min_dist, identity = model.predict("image.jpg")

            if min_dist > 0.8:
                st.write("## Sorry, you are not in the database")
            else:
                st.write(f"## Welcome: {identity}")
                print(f"identity, {identity} distance: {min_dist:.4f}")

# Create a Streamlit application
def main():
    st.title("Camera Capture and Image Classification")  
    captured_image = st.camera_input("Capture an image")
    if captured_image:
        predict_identity(captured_image)
        
# Run the application
if __name__ == "__main__":
    with st.spinner("Loading model ..."):
        model = FaceModel()
    main()