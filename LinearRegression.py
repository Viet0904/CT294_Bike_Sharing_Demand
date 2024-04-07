import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# pd.options.display.max_columns = 999

# Đọc dữ liệu từ file "train.csv"
df = pd.read_csv("BikeSharingDemand.csv")

# Chia dữ liệu thành X và y
X = df.drop(columns=["casual", "registered", "count"])
print(X)
y = (df["count"])  # Áp dụng logarithm cho cột "count"

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
# y_test = np.exp(y_test)
# y_pred = np.exp(y_pred)
# tinh score
score = model.score(X_test, y_test)
print("Score = ", score)
# Tính và in ra giá trị MSE và RMSE
print("MSE = ", mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))


# Tạo DataFrame mới từ dữ liệu test và kết quả dự đoán
# df_test = pd.DataFrame(
#     {
#         "season": X_test["season"],
#         "holiday": X_test["holiday"],
#         "workingday": X_test["workingday"],
#         "weather": X_test["weather"],
#         "temp": X_test["temp"],
#         "atemp": X_test["atemp"],
#         "humidity": X_test["humidity"],
#         "windspeed": X_test["windspeed"],
#         "Actual_Count": np.exp(y_test),
#         "Predicted_Count": np.exp(y_pred),
#     }
# )

# # Lưu DataFrame mới này vào tệp CSV
# df_test.to_csv("LinearRegression_predictions.csv", index=False)
