# Hindi-Digit-Recognizer-App

The "Hindi Handwritten Digit Recognition" project is a web application that allows users to upload photos of hindi handwritten digits or draw hindi digits directly on a canvas. It leverages machine learning techniques to predict the corresponding digit from the uploaded or drawn image.

## Functionality:

**Upload Images:** Users can upload images of handwritten digits in PNG, JPG, or JPEG format. The application provides a file uploader where users can select an image file from their local system.

**Image Display:** Once an image is uploaded, the application displays the uploaded image on the screen. The image is shown in its original color (BGR) format.

**Prediction:** Users can make a prediction on the uploaded image by clicking the "Make Prediction" button. The application uses a pre-trained convolutional neural network (CNN) model to predict the digit present in the image. The predicted digit label is then displayed to the user.

**Canvas Drawing: **The application provides a canvas component where users can draw digits directly. The canvas supports freehand drawing. Users can select the stroke width and stroke color for drawing.

**Canvas Prediction:** After drawing a digit on the canvas, users can click the "Make Prediction" button to make a prediction based on the drawn digit. The application converts the canvas image into the appropriate format and passes it to the pre-trained CNN model for prediction. The predicted digit label is displayed to the user.

**Image Saving:** When a prediction is made on the canvas image, the application saves the drawn image in the "output_images" directory. The saved image is in PNG format and is named with the predicted digit label and a timestamp. The file path of the saved image is displayed as a success message to the user.

## Technical Details:
The project utilizes the following libraries and technologies:

**PIL (Python Imaging Library):** Used to manipulate and process images.

**Streamlit:** A web application framework for building interactive applications.

**streamlit-drawable_canvas:** A Streamlit component for drawing on a canvas.

**OpenCV (cv2):** Used for image processing tasks, such as converting images between different color spaces.

**NumPy:** A library for numerical computing, used for array operations.

**Keras:** A deep learning library, used for loading and making predictions with the pre-trained CNN model.

**os:** Used for file system operations, such as creating directories and joining file paths.

**time:** Used for generating a timestamp for naming the saved images.

The application's interface consists of a heading displaying the project name, a file uploader for image upload, the uploaded image display, canvas for drawing, and buttons for making predictions. The application dynamically updates the displayed content based on user interactions.

By combining image processing techniques, machine learning, and an intuitive user interface, the "Hindi Handwritten Digit Recognition" project provides a convenient and interactive way to recognize and predict handwritten digits in the Hindi language.

## Running the app locally

1. Download the zip and extract the files in a folder.
2. Make sure you have streamlit installed, if not then open command prompt in the project folder and run the command "pip install streamlit".
3. Now, in the cmd run the command "streamlit run app.py".
4. The app will be opened in the browser.
