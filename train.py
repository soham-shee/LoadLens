# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import streamlit as st

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



def createGRUModel(df):
    prev=st.text_input('Enter the past lookup days : ')
    num=int(prev)
    training_set = list(df.values)
    training_set=training_set.reshape((training_set.shape[0],1))
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    Y_train = []
    for i in range(num,training_set.shape):
        X_train.append(training_set_scaled[i-num:i,0])
        Y_train.append(training_set_scaled[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)


st.title("Load Forecasting Using GRU")

df = st.file_uploader("Upload file", type={"csv"})
if df is not None:
    df = pd.read_csv(df, index_col=[0], parse_dates=[0])
    df=df['nat_demand'].resample('D').mean()
    st.write("Shape : ",df.shape)
    st.write("First 10 rows : ")
    st.write(df.head(10))
    fig=plt.figure(figsize=(16,6))
    plt.title('Graph of Net Demand vs Weekly data')
    plt.plot(df)
    st.pyplot(fig)
    # Main functions
    prev=int(st.text_input('Enter the past lookup days : '))
    training_set = list(df)
    training_set=training_set.reshape((training_set.shape[0],1))
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    Y_train = []
    for i in range(prev,training_set.shape):
        X_train.append(training_set_scaled[i-prev:i,0])
        Y_train.append(training_set_scaled[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
