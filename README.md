# Load Lens 😎

[![kaggle](https://camo.githubusercontent.com/0d9d4c150c1ea613d3bf3f89ea6f9323ed808b60ffef0ce7d942913aa33a256a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4b6167676c652d3230413545383f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6167676c65266c6f676f436f6c6f723d7768697465)](https://www.kaggle.com/code/sohamshee/gru-model-load-forecasting)
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
## Run Locally

Clone the project

```bash
  git clone https://github.com/soham-shee/LoadLens.git
```

Install dependencies

```bash
  pip install -r 'requirements.txt'
```

Start the server

```bash
  streamlit run App.py
```

To directly start it (Alternative Method)
```bash
  ./run_app.sh
```

## Demo

https://loadlens.streamlit.app/

## Acknowledgements

 - [Load Forecast Dataset (Panama Case Study)](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting)
