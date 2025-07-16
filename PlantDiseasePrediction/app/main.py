import PIL.Image
import streamlit as st
import numpy as np
import tensorflow as tf


import os
import json
import PIL

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_models/plant_disease_prediction_model.h5"

## Load the pre-trained model
model = tf.keras.models.load_model(model_path)

## Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

## Function to load and preprocess the image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = PIL.Image.open(image_path).convert('RGB')  # ✅ Force 3-channel
    img = img.resize(target_size, PIL.Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_disease(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions,axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


## Streamlit App
st.title("🪴 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_image is not None:
    image = PIL.Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((150, 150))
        st.image(resized_image)
    with col2:
        if st.button('Classify'):
            prediction = predict_disease(model, uploaded_image, class_indices)
            st.success(f"Prediction: {prediction}")
