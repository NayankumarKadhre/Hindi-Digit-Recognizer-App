from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import os
import time

# Define image size and number of channels
img_rows, img_cols, img_channels = 32, 32, 1
num_classes = 10

# Load the saved model
model = load_model('CNN_LSTM.h5')

# Function to preprocess the input image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image
    resized = cv2.resize(gray, (img_rows, img_cols))
    # Normalize the image
    normalized = resized.astype('float32') / 255.0
    # Expand dimensions to match the model input shape
    preprocessed = np.expand_dims(normalized, axis=0)
    preprocessed = np.expand_dims(preprocessed, axis=-1)
    return preprocessed

# Function to make predictions
def make_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Streamlit app
def main():

    st.markdown("<h1 style='text-align: center; color: white;'>Hindi Digit Recognizer</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image: ", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, channels="BGR", caption="Uploaded Image")
        if st.button("Make Prediction"):
            predicted_label = make_prediction(image)
            st.write(f"Predicted Label: {predicted_label}")

    # Specify canvas parameters in application
    drawing_mode = "freedraw"
    stroke_width = 11
    stroke_color = "#ffffff"
    bg_color = "#000000"
    # realtime_update = True

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if st.button("Make Prediction"):
        if canvas_result.image_data is not None:
            # Convert the canvas image to the appropriate format
            pil_image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Make prediction on the canvas image
            predicted_label = make_prediction(opencv_image)

            # Show the predicted label
            st.write(f"Predicted Label: {predicted_label}")

            # Save the drawn image
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            # Generate a unique filename using timestamp
            timestamp = int(time.time())
            filename = f"{predicted_label}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            pil_image.save(output_path)
            st.success(f"Canvas image saved at {output_path}")

if __name__ == '__main__':
    main()
