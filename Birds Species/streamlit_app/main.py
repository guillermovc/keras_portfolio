import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# A simple CNN model for demonstration purposes
# In a real scenario, this would be your pre-trained model
# Model needs to be initialized (for the demo, we simulate a model)
# For this example, let's simulate a CNN model that predicts 3 classes

@st.cache_resource
def load_model(model_path):
    # Load the Keras model from the specified path
    return tf.keras.saving.load_model(model_path)

# A simple CNN model with cached loading
class SimpleCNN:
    def __init__(self, model_path):
        # Load the cached model
        self.model = load_model(model_path)
        with open("classes.txt", "r") as class_file:
            classes = class_file.read()
            self.classes = classes.split("\n")

    def preprocess_image(self, image, target_size):
        # Resize, normalize, and expand dimensions for the model
        w, h = 224, 224
        img_array = np.array(image)

        center = img_array.shape
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2

        img_array = img_array[int(y):int(y+h), int(x):int(x+w)]
        img_array = np.expand_dims(img_array, axis=0)  # expand
        
        return img_array

    def predict(self, image):
        # Preprocess the image
        target_size = (224, 224)
        preprocessed_image = self.preprocess_image(image, target_size)
        with st.columns(3)[1]:
            st.image(preprocessed_image, "Preprocessed image")
        # Get the model's predictions
        predictions = self.model.predict(preprocessed_image)
        
        # Convert predictions to class labels
        predicted_class_index = np.argmax(predictions[0])
        
        return self.classes[predicted_class_index], np.max(predictions[0])
    

# Create a Streamlit application
def main():
    st.title("Camera Capture and Image Classification")  

    captured_image = st.camera_input("Capture an image")
    
    if captured_image:
        img = Image.open(captured_image)
        captured_image = np.array(img)
        center = captured_image.shape
        w, h = 224, 224
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2
        p1 = (int(x), int(y))
        p2 = (int(x+w), int(y+h))
        captured_image_zone = cv2.rectangle(captured_image.copy(), p1, p2, (255,0,0), thickness=3)

        st.image(captured_image_zone, caption="Captured Image (red zone will be cropped)", use_column_width=True)
        
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                prediction, score = model.predict(captured_image)
                st.write("Predicted Class:", prediction, f"{score:.2f}%")

# Run the application
if __name__ == "__main__":
    with st.spinner("Loading model ..."):
        model = SimpleCNN("custom_mobilenet_best_224.keras")
    main()