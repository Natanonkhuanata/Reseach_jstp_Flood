import tensorflow as keras
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



name_file = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
data = pd.read_excel(name_file)
data.columns = ['datetime' , 'rain' , 'Accumulated_rain' , 'Water_Level' , 'Water_quantity']
cols = [ 'rain' , 'Accumulated_rain' , 'Water_Level' , 'Water_quantity' ]
data[cols] = data[cols].astype(np.float32)
data['datetime'] = pd.to_datetime(data['datetime'])



num_data = int(len(data) * 0.70)

feature = ['rain' , 'Accumulated_rain' , 'Water_quantity']
target = ['Water_Level']

#Normalize data
scaler = MinMaxScaler()

x = data[feature].values
feature_scaler = scaler.fit_transform(x)

y= data[target].values
target_scaler = scaler.fit_transform(y)



x_train , x_test = feature_scaler[:num_data] , feature_scaler[num_data:]
y_train , y_test = target_scaler[:num_data] , target_scaler[num_data:]


# Sliding window
x_input = []
y_input = []

times_steps = 4  # one hour = 4 time steps
for i in range(times_steps, len(x_train)):
    x_input.append(x_train[i-times_steps:i])
    y_input.append(y_train[i])

x_input = np.array(x_input)
y_input = np.array(y_input)

x_input_test = []
y_input_test = []

for i in range(times_steps, len(x_test)):
    x_input_test.append(x_test[i-times_steps:i])
    y_input_test.append(y_test[i])

x_input_test = np.array(x_input_test)
y_input_test = np.array(y_input_test)


#print("shape x_input : " , x_input.shape)
#rint("shape y_input :" , y_input.shape)

# Reshape for LSTM
x_input = x_input.reshape(x_input.shape[0], x_input.shape[1], x_input.shape[2])
x_input_test = x_input_test.reshape(x_input_test.shape[0] , x_input_test.shape[1], len(feature))

#"""
# Load the prediction model
model = load_model("my_model_3_year_ago.keras")

y_pred = model.predict(x_input_test)

y_pred_inv = scaler.inverse_transform(y_pred)
print("prediction :", f"{y_pred_inv[0][0]:.2f}")

