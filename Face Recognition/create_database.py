import json
from pathlib import Path
import argparse

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np

#tf.keras.backend.set_image_data_format('channels_last')
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", help="Images folder", type=str)
parser.add_argument("-m", "--model", help="Model path", type=str)
parser.add_argument("-s", "--save", help="Path to save the json database", type=str)
args = parser.parse_args()

# Load model
json_file = open(Path(args.model) / 'model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(Path(args.model) / 'model.h5')

print("Model loaded ...")

# Load images
images = list(Path(args.images).glob("*"))

print(f"{len(images)} images loaded ...")
for img in images:
    print(img)

# Create database
database = {}
for img in images:
    person_name = img.name.split(".")[0].capitalize()
    
    database[person_name] = img_to_encoding(str(img), model).tolist()

# Save database
with open(args.save, 'w') as file:
    json.dump(database, file, indent=4)

print(f"Database created and stored in {args.save}")