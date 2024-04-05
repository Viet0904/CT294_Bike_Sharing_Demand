import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# pd.options.display.max_columns = 999
df = pd.read_csv("train.csv")
# Đọc dữ liệu từ file "train.csv" và lưu vào DataFrame df
# df = df[df["weather"] != 4]

# Tính giá trị trung bình của cột "windspeed"
# mean_windspeed = df["windspeed"].mean()
# print("Gia tri trung binh ", mean_windspeed)
# # Thay thế giá trị 0 bằng giá trị trung bình
# df["windspeed"] = df["windspeed"].replace(0, mean_windspeed)

# # Đếm số lượng giá trị của từng giá trị trong cột "windspeed"
# windspeed_counts = df["windspeed"].value_counts()

# Hiển thị các giá trị và số lượng xuất hiện của chúng
# print(windspeed_counts)
# Khởi tạo MinMaxScaler
# scaler = MinMaxScaler()
# # Rescale các thuộc tính temp, atemp và windspeed
# df[["temp", "atemp", "humidity", "windspeed","count"]] = scaler.fit_transform(
#     df[["temp", "atemp", "humidity", "windspeed","count"]]
# )

print(df)
# Chia dữ liệu thành X và y
X = df.drop(columns=["casual", "registered", "count", "datetime","atemp"])
print(X)
y = df["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train)
print(X_test)

# Xây dựng và huấn luyện mô hình Random Forest Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính và in ra giá trị MSE và RMSE
print("MSE = ", mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))
