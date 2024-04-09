import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Đọc dữ liệu từ file "train.csv" và lưu vào DataFrame df
df = pd.read_csv("train.csv")

# Tạo cột year, month, day, hour từ cột datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# Xóa cột "datetime" khỏi DataFrame df
df = df.drop(columns=["datetime"])

# Chia dữ liệu thành X và y
X = df.drop(columns=["casual", "registered", "count"])
y = np.log(df["count"])  # Áp dụng logarithm cho cột "count"

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Xây dựng và huấn luyện mô hình Decision Tree Regression
model = DecisionTreeRegressor()
model.fit(x_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(x_test)

# Tính và in ra giá trị MSE và RMSE
print("MSE = ", mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
