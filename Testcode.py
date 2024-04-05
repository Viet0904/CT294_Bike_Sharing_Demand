import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
df = pd.read_csv("train.csv")

# Chuẩn bị dữ liệu
df["datetime"] = pd.to_datetime(df["datetime"])

# Trích xuất các biến thời gian từ datetime
# df["hour"] = df["datetime"].dt.hour
# df["day"] = df["datetime"].dt.day
# df["month"] = df["datetime"].dt.month
# df["year"] = df["datetime"].dt.year
y = df["count"]
df.drop(columns=["count"], inplace=True)
df.drop(columns=["datetime"], inplace=True)
df.drop(columns=["casual"], inplace=True)
df.drop(columns=["registered"], inplace=True)
df.drop(columns=["atemp"], inplace=True)
print(df.head())
X = df
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)


# Dự đoán số lượng thuê xe đạp
y_pred = model.predict(X_test)
# Tính và in ra giá trị MSE và RMSE
print("MSE = ", mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
# In ra 10 dự đoán đầu tiên
print(y_pred[:10])
print(y_pred[-10:])
