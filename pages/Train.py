# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import streamlit as st
import pickle5 as pickle
import base64

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Demand')
    plt.plot(predicted, color='green',label='Predicted Demand')
    plt.title('Demand Prediction')
    plt.xlabel('Time')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()

# Calculation of Root Mean Squared-Error

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button_train():
    st.session_state.clicked = True



# def createGRUModel(df):
#     prev=st.text_input('Enter the past lookup days : ')
#     num=int(prev)
#     training_set = list(df.values)
#     training_set=training_set.reshape((training_set.shape[0],1))
#     sc = MinMaxScaler(feature_range=(0,1))
#     training_set_scaled = sc.fit_transform(training_set)
#     X_train = []
#     Y_train = []
#     for i in range(num,training_set.shape):
#         X_train.append(training_set_scaled[i-num:i,0])
#         Y_train.append(training_set_scaled[i,0])
#     X_train, Y_train = np.array(X_train), np.array(Y_train)

# st.markdown("""
#     <style>
#     body {
#         .center: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)
st.title("Load Forecasting Using GRU (Initial Training)")

df = st.file_uploader("Upload file", type={"csv"})
if df is not None:
    df = pd.read_csv(df, index_col=[0], parse_dates=[0])
    st.write("Shape : ",df.shape)
    st.write("First 10 rows : ")
    st.write(df.head(10))
    df=df['nat_demand'].resample('D').mean()
    values = st.slider(
    "Select a range for training values",
    2015, df.shape[0],(0,df.shape[0]//2))
    # st.write("Values:", values)
    start=values[0]
    end=values[1]
    # st.write(df.shape[0])
    # Ploting the graph
    fig=plt.figure(figsize=(16,6))
    plt.title('Graph of Net Demand vs Daily data')
    plt.plot(df[start:end])
    st.pyplot(fig)


    # Main functions
    df=df[start:end]
    prev = st.slider('Past lookup days (less than ending and starting dates) : ', key=2, min_value=1, max_value=100, value=1, step=1)
    training_set = np.array(df)
    training_set=np.reshape(training_set,(training_set.shape[0],1))
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    Y_train = []
    st.write(training_set_scaled.shape)
    for i in range(prev,training_set.shape[0]):
        X_train.append(training_set_scaled[i-prev:i,0])
        Y_train.append(training_set_scaled[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    col1, col2 = st.columns(2,gap="medium")
    with col1:
        st.header("X_train")
        st.write(X_train)
        st.write(X_train.shape)
    with col2:
        st.header("Y_train")
        st.write(Y_train)
        st.write(Y_train.shape)

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    # model
    EPOCHS = st.slider('Epochs : ', min_value=1, max_value=100, value=1, step=1)
    BATCH_SIZE = st.slider('Batch Size : ', min_value=1, max_value=100, value=1, step=1)

    model = Sequential()
    model.add(GRU(64, activation='tanh',return_sequences=True, input_shape=(60, 1)))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    #opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='adam', loss='mean_squared_error')
    if st.button("Train the Model", type="secondary", use_container_width=False):
        model.fit(X_train,Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)
        st.success("Model Trained Successfully", icon="🔥")

    def download_model(model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}" download="GRU_model.h5">Download Trained Model .h5 File</a>'
        st.markdown(href, unsafe_allow_html=True)


    download_model(model)
