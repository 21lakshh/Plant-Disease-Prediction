import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Get the absolute path of the file as the working directory may change
working_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = f"{working_dir}/trained_model/plant_disease_prediction.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

img_size = 224

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)  # Probability values for all classes
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]  # Convert index to class name
    confidence = prediction[0][predicted_class_index] * 100  # Confidence percentage
    return predicted_class_name, confidence

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Predicted Class: {prediction} \n\n Prediction %: {confidence:.2f}')
