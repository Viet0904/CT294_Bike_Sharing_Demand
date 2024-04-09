import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Đọc dữ liệu từ file CSV
data = pd.read_csv("./train.csv")

# Chuyển cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])

# Chọn các thuộc tính để train model và giá trị label
X = data[
    [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
]
y = data["count"]

# Chia dữ liệu thành train và test theo yêu cầu của bạn
train_data = data[data["datetime"].dt.day <= 15]
test_data = data[(data["datetime"].dt.day > 15)]

# In số lượng dòng của mỗi tập
print("Số lượng dòng trong tập train:", len(train_data))
print("Số lượng dòng trong tập test:", len(test_data))

# Chia dữ liệu thành X_train, X_test, y_train, y_test
X_train = train_data[
    [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
]
y_train = train_data["count"]
X_test = test_data[
    [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
]
y_test = test_data["count"]

# Xây dựng mô hình Random Forest Regressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Tính MSE và RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Ghi kết quả dự đoán vào file sampleSubmission
sample_submission = pd.DataFrame({"datetime": test_data["datetime"], "count": y_pred})
sample_submission.to_csv("RandomForestRegressor.csv", index=False)
