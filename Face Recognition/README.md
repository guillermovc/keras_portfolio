# Face Recognition System
This project implements a face recognition system using a trained neural network to recognize the same person by comparing embeddings generated from their photos. The system compares the embeddings of uploaded photos with a pre-existing database of embeddings to identify the person in the image.
  
The project includes a web application developed with Streamlit that runs inside a Docker container with TensorFlow, Python 3.6.9, TensorFlow 2.3.0, and Keras 2.4.0.

## Prerequisites
- Docker installed on your machine
- Images of the persons you want to recognize
- `model.json` and `model.h5` files containing the trained model
- `database.json` file containing the database embeddings or a folder with images

### Create container
Run:
`docker run -it --name tfcpu2 -p 8886:8886 -v ${PWD}:/keras_portfolio -w /keras_portfolio tensorflow/tensorflow:2.3.0-jupyter /bin/bash`

## How to run
- Inside the container launch the Streamlit app `streamlit run app.py --server.port 8886`.  
- The port should match the linked in the container.  
- Open your web browser and navigate to `http://localhost:8886` to access the application.