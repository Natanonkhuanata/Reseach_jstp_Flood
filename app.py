"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st

import pandas as pd
import numpy as np
import tensorflow as keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


name_file = "test.xlsx"
data = pd.read_excel(name_file)
data.columns = ['rain_fall', 'Accumulated_rain', 'Water_level', 'Water_quantity']
print(data)

features = ['Accumulated_rain', 'Water_quantity']
target = ['Water_level']
# แยกชุดข้อมูล
X = data[features].values
X_rain = data['rain_fall'].values.reshape(-1, 1)
X = X.reshape(-1, len(features)) # Reshape to ensure y is a 2D array
# Normalize features
features_scaler = MinMaxScaler()
X_scaled = features_scaler.fit_transform(X)
X_scaled = np.hstack((X_rain , X_scaled))

# Normalize target
y = data[target].values
y = y.reshape(-1, 1) # Reshape to ensure y is a 2D array
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)


X_test = X_scaled.reshape(-1, 10, X_scaled.shape[1]) # Reshape to 3D array for LSTM input


"""
print(features_test.shape)
print(target_test.shape)
print("\n")
print(features_test)
print(target_test)

"""

#"""
# Load the prediction model
model = load_model("my_model_3_year_ago.keras")

y_pred = model.predict(X_test)
print(y_pred)
print(y_pred.shape)

y_test_inv = target_scaler.inverse_transform(y_scaled)
y_pred_inv = target_scaler.inverse_transform(y_pred)
#print("prediction :", f"{y_pred_inv[0][0]:.2f}")

st.write(f"{y_pred_inv[0][0]:.2f}")