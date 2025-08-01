import pandas as pd
import numpy as np

from tensorflow import keras
from keras.models import load_model
import joblib

# โหลดโมเดล
model = load_model("C:\Github\opencv\Reseach_jstp_Flood\KingMungkud\model.h5")

# โหลด Scaler
scaler = joblib.load("C:\Github\opencv\Reseach_jstp_Flood\KingMungkud\scaler.pkl")


# ข้อมูลทดสอบ
Test = np.array([396.77 , 396.85 , 396.87 , 396.93 , 396.99 , 397.06 , 397.08 , 397.16 ,
                 397.29 , 397.24 , 397.29 , 397.37 , 397.37 , 397.36 , 397.34 , 397.36 ,
                 397.40 , 397.32 , 397.30 , 397.23 , 397.25 , 397.15 , 397.09 , 397.05])  # shape = (10,)

# Reshape เป็น 2D → (samples, features)
Test = Test.reshape(-1, 1)  # shape = (10, 1)



# ทำ scaling แบบทดลอง (ใช้ fit_transform ได้ในกรณีนี้)
Test_scaled = scaler.fit_transform(Test)

# แปลงกลับเป็น shape ที่ LSTM ต้องการ → (1, 10, 1)
Test_scaled = Test_scaled.reshape(1, 24, 1)

y_pred = model.predict(Test_scaled)
y_pred_inverese = scaler.inverse_transform(y_pred)


print("ผลการพยากรณ์ :" , y_pred_inverese)