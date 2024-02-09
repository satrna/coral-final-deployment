from roboflow import Roboflow
import os
import streamlit as st

def predict_image(image_file, conf, overlap):
    rf = Roboflow(api_key=st.secrets.roboflow_credentials.api_key)
    project = rf.workspace().project(st.secrets.roboflow_credentials.project_name)
    model = project.version(4).model
    file_path = os.path.join('storage', image_file)
    result = model.predict(file_path, confidence=conf, overlap=overlap).json()
    return result