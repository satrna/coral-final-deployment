import streamlit as st
from PIL import Image
import os
from utils.roboflow_utils import predict_image
import supervision as sv
import numpy as np


# Title
st.title('Coral Image Prediction')

# Confidence and Overlap Slider
confidence = st.slider("Confidence Threshold:", 10, 100, 50)
overlap = st.slider("Overlap Threshold:", 10, 100, 50)

# File Uploader
file_format = ['png', 'jpg']
uploaded_file = st.file_uploader("Choose a file", type=file_format)

# Predict Button
predict_button = st.button("Predict")

# Reset Button
reset_button = st.button("Reset", type="primary")

# label placeholder
labels = None

# Image Prediction Output
st.subheader('Coral Detection Output')
if predict_button:
    if uploaded_file is not None:
        with open('storage/' + uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        result = predict_image(uploaded_file.name, confidence, overlap)
        labels = {item["class"] for item in result["predictions"]}

        detections = sv.Detections.from_roboflow(result)
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        annotated_image = bounding_box_annotator.annotate(scene=image_np, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Display the annotated image
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
    else:
        st.write("Please upload an image first")

if reset_button:
    folder_path = "storage/"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.experimental_rerun()

# Text Inference From LLM About the Image
st.subheader('Coral Prediction Information')
if labels != {}:
    st.write(set(labels))
else:
    st.write("No Coral Detected")