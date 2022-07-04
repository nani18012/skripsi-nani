import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
##from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201,preprocess_input as densenet201_preprocess_input

model = tf.keras.models.load_model("saved_model/model_A4.hdf5")
### load file

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Identifikasi Penyakit Early Blight Daun Tomat')

uploaded_file = st.file_uploader("Upload File", type="jpg")

map_dict = {0: 'early_blight',
            1: 'sehat',
            2: 'penyakit_lainnya'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = densenet201_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Prediksi")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Hasil prediksi yaitu {}".format(map_dict [prediction]))