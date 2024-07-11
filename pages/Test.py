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
import pickle5 as pickle
import base64
from io import BytesIO

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
st.title("Load Forecasting Using GRU (Testing)")

df = st.file_uploader("Upload file", type={"csv"})
file = st.file_uploader('Model (.h5) file', type='.h5')
if df and file is not None:
    df = pd.read_csv(df, index_col=[0], parse_dates=[0])
    # model=pickle.load(open(file,'rb'))
    model_bytes = file.read()
    model = pickle.load(BytesIO(model_bytes))
    st.write("Shape : ",df.shape)
    st.write("First 10 rows : ")
    st.write(df.head(10))
    df=df['nat_demand'].resample('D').mean()
    values = st.slider(
    "Select a range for testing values",
    0, df.shape[0],(0,df.shape[0]-10))
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
    df_original=df
    df=df[start:end]
    prev = st.slider('Past lookup days (less than ending and starting dates) : ', key=2, min_value=1, max_value=200, value=1, step=1)
    test_set = np.array(df)
    test_set=np.reshape(test_set, (test_set.shape[0],1))
    num=test_set.shape[1]
    
    # After :
    # sc = MinMaxScaler(feature_range=(0,1))
    # dataset_total = pd.concat((df[:end],df[start:]),axis=0)
    # inputs = np.array(dataset_total[len(dataset_total)-len(test_set) - prev:])
    # inputs = inputs.reshape((-1,1))
    # inputs  = sc.fit_transform(inputs)

    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(test_set)
    inputs = df[len(test_set) - prev:].values
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    X_test = []
    for i in range(prev,prev+num):
        X_test.append(inputs[i-prev:i,0])
    # st.write(X_test[0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    predicted_data = model.predict(X_test)
    predicted_data = sc.inverse_transform(predicted_data)
    prediction=int(predicted_data[0])
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.write("Predicted Next Value : ")
        st.write(prediction)
    with col2:
        st.write("Original Next Value : ")
        st.write(df_original.iloc[end+1])

    # st.write(plot_predictions(test_set,predicted_data))
    # st.write(return_rmse(test_set,predicted_data))
