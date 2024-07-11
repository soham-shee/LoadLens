import streamlit as st
import time
import numpy as np
import pandas as pd

st.title("Load Forecasting Using Gated Recurrent Unit (GRU)")

_paragraph_ = """
This ML model is designed for load forecasting using Gated Recurrent Units (GRU). This user-friendly
app empowers users to input their past values data, specify the number of epochs, and set the 
batch size for training a GRU-based model. 
By leveraging the GRU architecture, the app efficiently captures temporal dependencies in the 
data, making it ideal for accurate load forecasting. Once the model is trained, users can easily 
download the trained model for future use, ensuring they have a reliable tool at their fingertips
for predicting load demand.

In addition to model training, this app offers a robust suite of features to enhance usability 
and flexibility. Users can upload a previously trained model alongside a CSV file to retrain 
the model, accommodating new data and improving prediction accuracy. This iterative approach 
ensures the model remains up-to-date with the latest trends and patterns. Furthermore, the app 
allows users to upload an existing model to forecast future values based on specified inputs, 
providing quick and precise predictions. Whether you are training a new model, retraining with 
additional data, or forecasting future values, this app offers a comprehensive solution for load 
forecasting needs.
"""


def stream_data():
    for word in _paragraph_.split(" "):
        yield word + " "
        time.sleep(0.02)

if st.button("See the description"):
    st.write_stream(stream_data)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    if st.button("Train Model (Initial Training)"):
        st.switch_page("pages/Train.py")
with col2:
    if st.button("Re-train Model"):
        st.switch_page("pages/Retrain.py")
with col3:
    if st.button("Testing Model"):
        st.switch_page("pages/Test.py")