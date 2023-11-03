import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests
model = tf.keras.models.load_model('potato_desices2.h5') 
# Define class names based on your dataset
class_names = ['Early_blight', 'Late_blight', 'healthy']  # Update with your class names

def predict_from_url_and_display(image_url):
    response = requests.get(image_url)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        st.write(f"Predicted class: {predicted_class}, Accuracy: {confidence}%")
        st.image(img, caption='Uploaded Image', use_column_width=True)

    else:
        st.error("Error fetching the image.")

# Streamlit app UI
st.title("Plant Disease Classifier")
st.write("Provide an image URL of 'Early blight', 'Late blight', and 'healthy' leaves of the plant to classify.")

st.markdown(
    """
    <p style="font-size: 18px;"><strong>You can provide image URLs of leaves for classification with Deep Learning😊</strong></p>
    
    <p style="font-size: 16px;">
    <ul>
        <li><strong>Example URL for 'Late blight':</strong> <a href="https://media.sciencephoto.com/b2/65/02/14/b2650214-800px-wm.jpg">Late Blight Leaf</a></li>
        <li><strong>Example URL for 'Early blight':</strong> <a href="https://assets.syngenta.ca/images/pest/125/ca2020_early_blight_potato_damage1.png">Early Blight Leaf</a></li>
        <li><strong>Example URL for 'Healthy':</strong> <a href="https://www.shutterstock.com/image-photo/potato-leaf-isolated-on-white-260nw-2299088565.jpg">Healthy Leaf</a></li>
    </ul>
    </p>
    
    <p style="font-size: 14px;"><strong>Please make sure to provide clear and distinct images for accurate classification.</strong></p>
    """,
    unsafe_allow_html=True
)

image_url = st.text_input("Enter Image URL:")

if st.button("Classify from URL"):
    if image_url:
        predict_from_url_and_display(image_url)
