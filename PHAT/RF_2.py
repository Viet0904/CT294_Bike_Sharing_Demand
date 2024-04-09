import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

# Đọc dữ liệu từ tập tin CSV
dt = pd.read_csv('C:/Users/tuanp/Downloads/train.csv')

dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["year"] = dt["datetime"].dt.year
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour

X = dt.drop(columns=["casual", "registered", "count", "datetime"])
print(X)
y = dt["count"]
print(y)
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)
print(X_train)
print(X_test)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)
# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")