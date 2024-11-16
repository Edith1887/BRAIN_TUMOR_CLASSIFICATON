import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('D:/BRAIN TUMOR PREDICTION/brain_tumor_model.h5')

# Define the class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize image to 150x150
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit app interface
st.title('Brain Tumor Prediction')
st.write('Upload an image to predict if there is a brain tumor or not.')

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image using PIL
    from PIL import Image
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    predicted_prob = predictions[0][predicted_class_index]

    # Show prediction result
    st.write(f"The model predicts the image as: **{predicted_class}**")
    st.write(f"Probability: **{predicted_prob * 100:.2f}%**")

    # Display the image with prediction result
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f'Predicted: {predicted_class} (Probability: {predicted_prob*100:.2f}%)')
    ax.axis('off')
    st.pyplot(fig)
