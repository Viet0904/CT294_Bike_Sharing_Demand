import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

dt = pd.read_csv("train.csv")

print("Hiển thị 5 dòng dữ liệu đầu tiên của tập dữ liệu\n", dt.head())
print("Hiển thị thông tin của tập dữ liệu\n", dt.info())
print("Hiển thị số lượng dòng và cột của tập dữ liệu\n", dt.shape)
print("Tóm tắt thống kê của tập dữ liệu\n", dt.describe())

# Xoá các cột
dt = dt.drop(columns=["casual", "registered", "datetime"])
print(
    "Hiển thị 5 dòng dữ liệu đầu tiên của tập dữ liệu sau khi xóa cột 'casual', 'registered', 'datetime'\n",
    dt.head(),
)

# Kiểm tra các giá trị thiếu
print("Kiểm tra giá trị thiếu trong tập dữ liệu\n", dt.isnull().sum())

# Đổi tên cột
codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
dt["season"] = dt["season"].map(codes)
sns.barplot(x="season", y="count", data=dt)
plt.show()
# chuyển ngược lại
codes = {"spring": 1, "summer": 2, "fall": 3, "winter": 4}
dt["season"] = dt["season"].map(codes)


codes = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
dt["weather"] = dt["weather"].map(codes)
sns.barplot(x="weather", y="count", data=dt)
plt.show()
# chuyển ngược lại
codes = {"Clear": 1, "Mist": 2, "Light Snow": 3, "Heavy Rain": 4}
dt["weather"] = dt["weather"].map(codes)


codes = {1: "working_day", 0: "Holiday"}
dt["workingday"] = dt["workingday"].map(codes)
sns.barplot(x="workingday", y="count", data=dt, palette="cool")
plt.show()
# chuyển ngược lại
codes = {"working_day": 1, "Holiday": 0}
dt["workingday"] = dt["workingday"].map(codes)

plt.scatter(x="temp", y="count", data=dt)
plt.show()

plt.scatter(x="atemp", y="count", data=dt)
plt.show()
plt.scatter(x="humidity", y="count", data=dt)
plt.show()
plt.scatter(x="windspeed", y="count", data=dt)
plt.show()

sns.distplot(dt["count"])
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(dt.corr(), annot=True)
plt.show()

data = dt[["temp", "atemp", "humidity", "windspeed"]]
sns.heatmap(data.corr(), annot=True)
plt.show()


