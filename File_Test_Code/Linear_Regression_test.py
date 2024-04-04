import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import math

# Đọc dữ liệu từ file CSV
data = pd.read_csv("./train.csv")

# Chuyển đổi cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])
print("Dữ liệu trước khi xử lý")
print("Shape of data:", data.shape)
print("5 dòng đầu của tập dữ liệu:\n", data.head())
# xoá cột 'casual' và 'registered'
data = data.drop(["casual", "registered"], axis=1)
# Tạo các tính năng mới từ cột 'datetime'
data["hour"] = data["datetime"].dt.hour
data["dayofweek"] = data["datetime"].dt.dayofweek
data["month"] = data["datetime"].dt.month
data["year"] = data["datetime"].dt.year
print("\nDữ liệu sau khi xử lý")
print("Shape of data:", data.shape)
print("5 dòng đầu của tập dữ liệu:\n", data.head())

# vẽ đồ thị biểu diễn số lượng xe đạp theo giờ trong ngày
import matplotlib.pyplot as plt

plt.plot(data.groupby("hour")["count"].mean())
plt.xlabel("Hour")
plt.ylabel("Count")
plt.title("Count vs Hour")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo tháng trong năm
plt.plot(data.groupby("month")["count"].mean())
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Count vs Month")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo năm
plt.plot(data.groupby("year")["count"].mean())
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Count vs Year")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo ngày trong tuần
plt.plot(data.groupby("dayofweek")["count"].mean())
plt.xlabel("Day of week")
plt.ylabel("Count")
plt.title("Count vs Day of week")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo mùa
plt.plot(data.groupby("season")["count"].mean())
plt.xlabel("Season")
plt.ylabel("Count")
plt.title("Count vs Season")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo thời tiết
plt.plot(data.groupby("weather")["count"].mean())
plt.xlabel("Weather")
plt.ylabel("Count")
plt.title("Count vs Weather")
plt.show()

# vẽ đồ thị biểu diễn số lượng xe đạp theo ngày lễ
plt.plot(data.groupby("holiday")["count"].mean())
plt.xlabel("Holiday")
plt.ylabel("Count")
plt.title("Count vs Holiday")
plt.show()


# Chia tập dữ liệu thành train và test
train = data[data["datetime"].dt.day <= 15]
test = data[(data["datetime"].dt.day > 15) & (data["datetime"].dt.day <= 19)]
print("\nDữ liệu sau khi chia")
print("Dữ liệu tập train")
print("Shape of Train ", train.shape)
print("5 dòng đầu của tập train:\n", train.head())
print("Thống kê trên tập train\n", train.describe())
print("\nDữ liệu tập test")
print("Shape of Test ", test.shape)
print("5 dòng đầu của tập test:\n", test.head())
print("Thống kê trên tập test\n", test.describe())


# Chọn các đặc trưng để huấn luyện mô hình
features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "dayofweek",
    "month",
    "year",
]

# Xây dựng mô hình hồi quy tuyến tính trên tập train
model = LinearRegression()
model.fit(train[features], train["count"])
# Gia tri theta(i) (i from 1 to n voi n la so luong thuoc tinh)
print(model.coef_)
# Gia tri theta0
print(model.intercept_)
# Du doan gia nha cho tap test
y_pred = model.predict(test[features])
# Đọc dữ liệu từ file sampleSubmission
sample_submission = pd.read_csv("./sampleSubmission.csv")
# Cập nhật các giá trị dự đoán vào tệp sampleSubmission
sample_submission["count"] = y_pred

# Lưu tệp sampleSubmission đã được cập nhật
sample_submission.to_csv("./sampleSubmission_predicted.csv", index=False)

err = mean_squared_error(test["count"], y_pred)
print("MSE =", str(err))
rmse_err = math.sqrt(err)
print("RMSE =" + str(round(rmse_err, 3)))
# Dự đoán trên tập test
predictions = model.predict(test[features])



# Đọc dữ liệu từ tệp sampleSubmission đã được dự đoán
predicted_data = pd.read_csv("./sampleSubmission_predicted.csv")


# Vẽ đồ thị biểu diễn số lượng xe đạp dự đoán và thực tế trên tập test với dấu chấm
plt.plot(test["datetime"], test["count"], marker='o', linestyle='', label="Thực tế")
plt.plot(test["datetime"], predicted_data["count"], marker='o', linestyle='', label="Dự đoán")
plt.xlabel("Thời gian")
plt.ylabel("Số lượng xe đạp")
plt.title("So sánh số lượng xe đạp dự đoán và thực tế")
plt.xticks(rotation=45)
plt.legend()
plt.show()
