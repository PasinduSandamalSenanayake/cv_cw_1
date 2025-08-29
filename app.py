import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from PIL import Image
import h5py

# Load the model architecture from the .h5 file
with h5py.File("my_model.h5", "r") as f:
    model_json = f.attrs.get("model_config")  # get model structure

# Rebuild the model from JSON
model = model_from_json(model_json)

# Load the trained weights
model.load_weights("my_model.h5")

# Compile the model (needed for prediction)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define class names
class_names = ['plastic', 'organic', 'metal']

# Get the expected input shape of the model
input_shape = model.input_shape 
st.write(f"Model expects input of shape: {input_shape}")

# Streamlit App UI
st.title("Image Classification App")
st.write("Upload an image and the model will predict its category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and convert the image to RGB
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize image to match model's input
    target_size = (input_shape[1], input_shape[2])
    img = img.resize(target_size)

    # Convert image to numpy array
    img_array = image.img_to_array(img)

    # Add batch dimension (model expects batch input)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (scale pixel values to 0-1)
    img_array = img_array / 255.0

    # Convert numpy array to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Make prediction
    prediction = model.predict(input_tensor)

    # Get predicted class label
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.success(f"Predicted class: **{predicted_class}**")
