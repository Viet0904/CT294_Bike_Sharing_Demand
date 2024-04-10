import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

dt = pd.read_csv("./BikeSharingDemand.csv")
print(dt.isnull().sum())
# tạo cột year, month, day, hour từ cột datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["hour"] = dt["datetime"].dt.hour
dt["year"] = dt["datetime"].dt.year
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


# Tạo một hình mới với kích thước cố định
plt.figure(figsize=(15, 10))

# Đồ thị cho mùa
plt.subplot(2, 3, 1)
codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
dt["season"] = dt["season"].map(codes)
sns.barplot(x="season", y="count", data=dt)
plt.title("Count vs Season")

# Đồ thị cho ngày lễ
plt.subplot(2, 3, 2)
codes = {1: "holiday", 0: "not holiday"}
dt["holiday"] = dt["holiday"].map(codes)
sns.barplot(x="holiday", y="count", data=dt, palette="cool")
plt.title("Count vs Holiday")

# Đồ thị cho ngày làm việc
plt.subplot(2, 3, 3)
codes = {1: "working day", 0: "not workingday"}
dt["workingday"] = dt["workingday"].map(codes)
sns.barplot(x="workingday", y="count", data=dt, palette="cool")
plt.title("Count vs Working Day")

# Đồ thị cho thời tiết
plt.subplot(2, 3, 4)
codes = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
dt["weather"] = dt["weather"].map(codes)
sns.barplot(x="weather", y="count", data=dt)
plt.xticks(rotation=90)
plt.title("Count vs Weather")

# Đồ thị cho nhiệt độ
plt.subplot(2, 3, 5)
plt.scatter(x=dt["temp"], y=dt["count"])
plt.xlabel("temp")
plt.ylabel("count")
plt.title("Scatter plot of Count vs Temperature")

# Đồ thị cho nhiệt độ cảm nhận
plt.subplot(2, 3, 6)
plt.scatter(x="atemp", y="count", data=dt)
plt.xlabel("atemp")
plt.ylabel("count")
plt.title("Scatter plot of Count vs Atemp")

# Hiển thị hình
plt.tight_layout()
plt.show()

# Tạo một hình mới với kích thước cố định
plt.figure(figsize=(18, 10))

# Đồ thị cho độ ẩm
plt.subplot(2, 3, 1)
plt.scatter(x="humidity", y="count", data=dt)
plt.xlabel("humidity")
plt.ylabel("count")
plt.title("Scatter plot of Count vs Humidity")

# Đồ thị cho tốc độ gió
plt.subplot(2, 3, 2)
plt.scatter(x="windspeed", y="count", data=dt)
plt.xlabel("windspeed")
plt.ylabel("count")
plt.title("Scatter plot of Count vs Windspeed")

# Đồ thị cho ngày trong tuần
plt.subplot(2, 3, 3)
code = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
dt["weekday"] = dt["weekday"].map(code)
sns.barplot(x="weekday", y="count", data=dt)
plt.xticks(rotation=90)
plt.title("Count vs Weekday")

# Đồ thị cho giờ trong ngày
plt.subplot(2, 3, 4)
sns.barplot(x="hour", y="count", data=dt)
plt.title("Count vs Hour")

# Đồ thị cho tháng trong năm
plt.subplot(2, 3, 5)
code = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
dt["month"] = dt["month"].map(code)
sns.barplot(x="month", y="count", data=dt)
plt.xticks(rotation=90)
plt.title("Count vs Month")


# Đồ thị cho năm
plt.subplot(2, 3, 6)
sns.barplot(x="year", y="count", data=dt)
plt.title("Count vs Year")


# Hiển thị hình
plt.tight_layout()
plt.show()
