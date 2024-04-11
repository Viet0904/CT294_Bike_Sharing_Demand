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
dt["year"] = dt["datetime"].dt.year
dt["hour"] = dt["datetime"].dt.hour
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
# Trực quan hoá dữ liệu count và season
codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
dt["season"] = dt["season"].map(codes)
sns.barplot(x="season", y="count", data=dt)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# Kiểm tra các giá trị thiếu
print("Kiểm tra giá trị thiếu trong tập dữ liệu\n", dt.isnull().sum())
# Trực quan hoá dữ liệu count và year
sns.barplot(x="year", y="count", data=dt)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# Trực quan hoá dữ liệu count và month
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
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
code = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
dt["month"] = dt["month"].map(code)

# Trực quan hoá dữ liệu count và weekday
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
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
code = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
dt["weekday"] = dt["weekday"].map(code)

# Trực quan hoá dữ liệu count và hour
sns.barplot(x="hour", y="count", data=dt)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# Trực quan hoá dữ liệu count và season
codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
dt["season"] = dt["season"].map(codes)
sns.barplot(x="season", y="count", data=dt)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
codes = {"spring": 1, "summer": 2, "fall": 3, "winter": 4}
dt["season"] = dt["season"].map(codes)

# Trực quan hoá dữ liệu count và weather
codes = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
dt["weather"] = dt["weather"].map(codes)
sns.barplot(x="weather", y="count", data=dt)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
codes = {"Clear": 1, "Mist": 2, "Light Snow": 3, "Heavy Rain": 4}
dt["weather"] = dt["weather"].map(codes)

# Trực quan hoá dữ liệu count và workingday
codes = {1: "working day", 0: "not workingday"}
dt["workingday"] = dt["workingday"].map(codes)
sns.barplot(x="workingday", y="count", data=dt, palette="cool")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
codes = {"working_day": 1, "not workingday": 0}
dt["workingday"] = dt["workingday"].map(codes)

# trực quan hoa dữ liệu count và holiday
codes = {1: "holiday", 0: "not holiday"}
dt["holiday"] = dt["holiday"].map(codes)
sns.barplot(x="holiday", y="count", data=dt, palette="cool")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# chuyển ngược lại
codes = {"holiday": 1, "not holiday": 0}
dt["holiday"] = dt["holiday"].map(codes)

# Trực quan hoá dữ liệu count và temp
plt.scatter(x=dt["temp"], y=dt["count"])
plt.xlabel("temp")
plt.ylabel("Count")
plt.title("Scatter plot of Count vs Temperature")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# Trực quan hoá dữ liệu count và atemp
plt.scatter(x="atemp", y="count", data=dt)
plt.xlabel("atemp")
plt.ylabel("Count")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# Trực quan hoá dữ liệu count và humidity
plt.scatter(x="humidity", y="count", data=dt)
plt.xlabel("humidity")
plt.ylabel("Count")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# Trực quan hoá dữ liệu count và windspeed
plt.scatter(x="windspeed", y="count", data=dt)
plt.xlabel("windspeed")
plt.ylabel("Count")
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
