import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Đọc dữ liệu từ file CSV
data = pd.read_csv("./train.csv")

# Chuyển cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])

# Chọn các thuộc tính để train model và giá trị label
X = data[
    [
        "datetime",  # Bao gồm cột datetime
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

# Chia dữ liệu thành train và test theo tỷ lệ 2/3 - 1/3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train.drop("datetime", axis=1), y_train)  # Loại bỏ cột datetime trong quá trình huấn luyện

# Dự đoán trên tập test
y_pred = model.predict(X_test.drop("datetime", axis=1))  # Loại bỏ cột datetime trong quá trình dự đoán

# Tính MSE và RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE là căn bậc hai của MSE

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Ghi kết quả dự đoán vào file sampleSubmission
sample_submission = pd.DataFrame({"datetime": X_test["datetime"], "count": y_pred})
sample_submission.to_csv("LinearRegression.csv", index=False)
