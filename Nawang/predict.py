import pandas as pd
import numpy as np

from tensorflow import keras
from keras.models import load_model
import joblib



# โหลดโมเดล
model = load_model("C:/Github/opencv/Reseach_jstp_Flood/Nawang/model.h5")

# โหลด Scaler
scaler = joblib.load("C:/Github/opencv/Reseach_jstp_Flood/Nawang/scaler.pkl")

# สมมุติว่าคุณมีข้อมูลใหม่
# ข้อมูลทดสอบ
Test = np.array([442.18 , 442.11 , 442.03 , 441.96 , 441.90])  # shape = (10,)







"""  not  edit  """

# Reshape เป็น 2D → (samples, features)
Test = Test.reshape(-1, 1)  # shape = (10, 1)


# ทำ scaling แบบทดลอง (ใช้ fit_transform ได้ในกรณีนี้)
Test_scaled = scaler.fit_transform(Test)


# แปลงกลับเป็น shape ที่ LSTM ต้องการ → (1, 10, 1)
Test_scaled = Test_scaled.reshape(1, 5, 1)


y_pred = model.predict(Test_scaled)
y_pred_inverese = scaler.inverse_transform(y_pred)


print("ผลการพยากรณ์ :" , y_pred_inverese[0])
