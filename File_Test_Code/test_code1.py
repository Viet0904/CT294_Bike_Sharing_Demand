import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Đọc dữ liệu từ file CSV vào DataFrame
data = pd.read_csv("./train.csv")
# Chuyển cột 'datetime' sang định dạng datetime
data["datetime"] = pd.to_datetime(data["datetime"])
# Chia tập train và test
train_data = data[data["datetime"].dt.day <= 15]
test_data = data[data["datetime"].dt.day > 15]
# In số lượng dòng của mỗi tập
print("Số lượng dòng trong tập train:", len(train_data))
print("Số lượng dòng trong tập test:", len(test_data))
# Lưu tập train và test vào các file CSV
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)


print("Dữ liệu trước khi xử lý")
train = pd.read_csv("./train_data.csv", parse_dates=["datetime"])
print(train.shape)
test = pd.read_csv("./test_data.csv", parse_dates=["datetime"])
print(test.shape)


# Các feature liên tục và feature hạng mục
# Feature liên tục = ["temp","humidity","windspeed","atemp"]
# Chuyển đổi kiểu của feature hạng mục sang kiểu category
categorical_feature_names = [
    "season",
    "holiday",
    "workingday",
    "weather",
]

for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

feature_names = [
    "season",
    "weather",
    "temp",
    "atemp",
    "workingday",
    "humidity",
    "holiday",
    "windspeed",
]

print("Các thuộc tính", feature_names)
X_train = train[feature_names]

print("Giá trị của X_train")
print(X_train.shape)
print(X_train.head())

print("Giá trị của X_test")
X_test = test[feature_names]
print(X_test.shape)
print(X_test.head())

label_name = "count"
y_train = train[label_name]
print("Giá trị của y_train")
print(y_train.shape)
print(y_train.head())

y_test = test[label_name]
print("Giá trị của y_test")
print(y_test.shape)
print(y_test.head())


# SkLearn SGD classifier
lm = SGDRegressor(max_iter=100)
lm.fit(X_train, y_train)
plt.scatter(y_test, lm.predict(X_test))
plt.grid()
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("scatter plot between actual y and predicted y")
plt.show()


predictions = lm.predict(X_test)

submission = pd.read_csv("./sampleSubmission.csv")
print(submission)

# Điều chỉnh độ dài của dự đoán để phù hợp với độ dài chỉ mục của DataFrame
predictions_adjusted = predictions[: len(submission)]

# Gán giá trị dự đoán vào DataFrame
submission["count"] = predictions_adjusted

# In kết quả
print(submission.shape)
submission.head()

submission.to_csv("sampleSubmission_predicted", index=False)

print("Mean Squared Error :", mean_squared_error(y_test, lm.predict(X_test)))
print("Mean Absolute Error :", mean_absolute_error(y_test, lm.predict(X_test)))
print("score ", SGDRegressor.score(y_test,predictions))