import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = r"D:\Study\Graduation_Project\Data_set\Inception model\Evaluations and visuals\best_inception_model_7.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Baryte', 'Calcite', 'Fluorite', 'Pyrite']

# Additional information about each rock
ROCK_INFO = {
    "Baryte": "Baryte is a mineral consisting of barium sulfate. It is generally white or colorless and is the main source of barium.",
    "Calcite": "Calcite is a carbonate mineral and the most stable polymorph of calcium carbonate (CaCO3). It is often transparent or white.",
    "Fluorite": "Fluorite is a colorful mineral, both in visible and ultraviolet light, and the stone has ornamental and industrial uses.",
    "Pyrite": "Pyrite, also known as fool's gold, is a sulfide mineral with a metallic luster and pale brass-yellow hue."
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((450, 450))  # Resize to match the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the class
def predict_image(image):
    predictions = model.predict(image)[0]
    max_prob = np.max(predictions)
    class_idx = np.argmax(predictions)

    if max_prob < 0.75:  # Confidence threshold
        return "Image not recognized", None
    else:
        return CLASS_LABELS[class_idx], max_prob

# Streamlit UI
st.title("Rock Classification App")
st.write("Upload an image of a rock, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Upload an image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    predicted_class, confidence = predict_image(preprocessed_image)

    # Display prediction
    if confidence is None:
        st.write("### Result: Image not recognized")
    else:
        st.write(f"### Result: **{predicted_class}**")
        st.write(f"### Confidence: {confidence * 100:.2f}%")

        # Display additional information
        st.markdown("#### About the predicted rock:")
        st.write(ROCK_INFO.get(predicted_class, "No information available for this rock."))
