import pylab
import calendar
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

# Tiền xử lý dữ liệu
train_data = pd.read_csv("./train_data.csv")

"""
    Tiền xử lý dữ liệu bằng cách  đọc file CSV, chuyển cột datetime thành đối tượng datetime,
    thêm các cột mới, loại bỏ các cột không cần thiết và hiển thị thông tin về dữ liệu.
"""
print("Hiển thị 5 dòng đầu tiên của tập Train")
print(train_data.head())
print("Thông tin về tập Train")
print(train_data.info())
print("Số lượng giá trị thiếu trong tập Train")
print(train_data.isnull().sum())
print("Các thống kê mô tả của tập Train")
print(train_data.describe())
train_data["datetime"] = pd.to_datetime(train_data["datetime"], format="%m/%d/%Y %H:%M")
train_data["hour"] = train_data["datetime"].dt.hour
train_data["month"] = train_data["datetime"].dt.month
train_data["weekday"] = train_data["datetime"].dt.weekday
train_data["year"] = train_data["datetime"].dt.year
train_data.drop(["casual", "registered"], axis=1, inplace=True)
print("Số lượng giá trị duy nhất trong tập Train")
print(train_data.nunique())
print("Hiển thị 5 dòng đầu tiên của tập Train")
print(train_data.head())


# Hiển thị 5 dòng đầu tiên của DataFrame
print(train_data.head())

# Biểu đồ và trực quan hoá dữ liệu
fig = plt.figure(figsize=[12, 10])
ax1 = fig.add_subplot(2, 2, 1)
ax1 = sns.barplot(
    data=train_data.groupby("year")["count"].mean().reset_index(), x="year", y="count"
)
ax2 = fig.add_subplot(2, 2, 2)
ax2 = sns.barplot(
    data=train_data.groupby("season")["count"].mean().reset_index(),
    x="season",
    y="count",
)
ax3 = fig.add_subplot(2, 2, 3)
ax3 = sns.barplot(
    data=train_data.groupby("weather")["count"].mean().reset_index(),
    x="weather",
    y="count",
)
ax4 = fig.add_subplot(2, 2, 4)
ax4 = sns.barplot(
    data=train_data.groupby("holiday")["count"].mean().reset_index(),
    x="holiday",
    y="count",
)

fig = plt.figure(figsize=[12, 10])
ax1 = fig.add_subplot(2, 2, 1)
ax1 = sns.barplot(
    data=train_data.groupby("workingday")["count"].mean().reset_index(),
    x="workingday",
    y="count",
)
ax2 = fig.add_subplot(2, 2, 2)
ax2 = sns.barplot(
    data=train_data.groupby("month")["count"].mean().reset_index(), x="month", y="count"
)
ax3 = fig.add_subplot(2, 2, 3)
ax3 = sns.barplot(
    data=train_data.groupby("weekday")["count"].mean().reset_index(),
    x="weekday",
    y="count",
)
ax4 = fig.add_subplot(2, 2, 4)
ax4 = sns.barplot(
    data=train_data.groupby("hour")["count"].mean().reset_index(), x="hour", y="count"
)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.flatten()
fig.set_size_inches(12, 20)

hour_season = pd.DataFrame(
    train_data.groupby(["hour", "season"], sort=True)["count"].mean()
).reset_index()
hour_season["seasonname"] = hour_season.season.map(
    {1: "Spring", 2: "Summer", 3: "autumn", 4: "Winter"}
)
sns.pointplot(
    x=hour_season["hour"],
    y=hour_season["count"],
    hue=hour_season["seasonname"],
    data=hour_season,
    join=True,
    ax=ax1,
)
ax1.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count By Hour Of The Day Across Weekdays",
    label="big",
)

hour_weekday = pd.DataFrame(
    train_data.groupby(["hour", "weekday"], sort=True)["count"].mean()
).reset_index()
hour_weekday["weekdayname"] = hour_weekday.weekday.apply(lambda x: calendar.day_name[x])
sns.pointplot(
    x=hour_weekday["hour"],
    y=hour_weekday["count"],
    hue=hour_weekday["weekdayname"],
    data=hour_weekday,
    join=True,
    ax=ax2,
)
ax2.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count By Hour Of The Day Across Weekdays",
    label="big",
)

hour_holiday = pd.DataFrame(
    train_data.groupby(["hour", "holiday"], sort=True)["count"].mean()
).reset_index()
hour_holiday["holiday_map"] = hour_holiday.holiday.map({0: "holiday", 1: "working day"})
sns.pointplot(
    x=hour_holiday["hour"],
    y=hour_holiday["count"],
    hue=hour_holiday["holiday_map"],
    data=hour_holiday,
    join=True,
    ax=ax3,
)
ax3.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count By Hour Of The Day Across Weekdays",
    label="big",
)
fig.subplots_adjust(hspace=0.5, wspace=0.5)


corr = train_data.drop(["datetime"], axis=1).corr()
plt.figure(figsize=(25, 25))
sns.heatmap(
    corr,
    cbar=True,
    square=True,
    fmt=".1f",
    annot=True,
    annot_kws={"size": 15},
    cmap="YlGnBu",
    linewidths=1,
)
plt.xticks(rotation=90)  # Xoay các nhãn trên trục x
plt.yticks(rotation=0)  # Xoay các nhãn trên trục y để chúng nằm ngang

select_features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "weekday",
    "month",
    "year",
    "hour",
]
X_train = train_data[select_features]
y_train = train_data["count"]

# Huấn luyện mô hình Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

test_data = pd.read_csv("./test_data.csv")
test_data["datetime"] = pd.to_datetime(test_data["datetime"], format="%m/%d/%Y %H:%M")
test_data["hour"] = test_data["datetime"].dt.hour
test_data["month"] = test_data["datetime"].dt.month
test_data["weekday"] = test_data["datetime"].dt.weekday
test_data["year"] = test_data["datetime"].dt.year
test_datetime = test_data["datetime"]
test_data.drop(["datetime"], axis=1, inplace=True)
print("Hiển thị 5 dòng đầu tiên của tập Test")
print(test_data.head())

test_predict = lr.predict(test_data[select_features])


X_train = train_data[select_features]
y_train = train_data["count"]

def print_evaluate(true, predicted):
    """
    In kết quả đánh giá của mô hình đã huấn luyện.
    """
    print("=======Kết quả đánh giá mô hình huấn luyện=======")
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print("RMSE:", rmse)
    print("R2 Square:", r2_square)

lr.fit(X_train, y_train)
print_evaluate(y_train, lr.predict(X_train))

# Biểu diễn kết quả dự đoán trên tập test
plt.figure(figsize=(10, 6))
plt.plot(test_datetime, test_predict, label="Predicted Count", color="red")
plt.title("Predicted Count on Test Data using Linear Regression")
plt.xlabel("Datetime")
plt.ylabel("Count")
plt.legend()
plt.show()
