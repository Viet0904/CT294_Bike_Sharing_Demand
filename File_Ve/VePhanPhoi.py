import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

dt = pd.read_csv("./BikeSharingDemand.csv")

print(dt.dtypes)
print("\n\n\n")
print("Hiển thị 5 dòng dữ liệu đầu tiên của tập dữ liệu\n", dt.head())
print("Hiển thị thông tin của tập dữ liệu\n", dt.info())
print("Hiển thị số lượng dòng và cột của tập dữ liệu\n", dt.shape)
print("Tóm tắt thống kê của tập dữ liệu\n", dt.describe())
# Tạo cột hour, month, weekday, year từ dộ datetime
dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["hour"] = dt["datetime"].dt.hour
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["year"] = dt["datetime"].dt.year

# print(dt["year"])
dt.drop("datetime", axis=1, inplace=True)
dt.drop("casual", axis=1, inplace=True)
dt.drop("registered", axis=1, inplace=True)

print("\n\n\n")
print(dt.dtypes)
print("Hiển thị 5 dòng dữ liệu đầu tiên của tập dữ liệu\n", dt.head())
print("Hiển thị thông tin của tập dữ liệu\n", dt.info())
print("Hiển thị số lượng dòng và cột của tập dữ liệu\n", dt.shape)
print("Tóm tắt thống kê của tập dữ liệu\n", dt.describe())
# plt.figure(figsize=(12, 6))
# sns.heatmap(dt.corr(), annot=True, cmap="coolwarm")
# plt.title("Heatmap of the dataset")
# plt.xticks(rotation=0)
# plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
# plt.show()


# vẽ biểu đồ phân phối của từng cột
# vẽ biểu đồ phân phối của cột hour
plt.figure(figsize=(12, 6))
plt.title("Distribution of hour")
plt.hist(dt["hour"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("hour")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()


# vẽ biểu đồ phân phối của cột year
plt.figure(figsize=(12, 6))
plt.title("Distribution of year")
plt.hist(dt["year"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("year")
plt.ylabel("Frequency")
plt.xticks([2011, 2012])
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
# Vẽ biểu đồ phân phối của cột month
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
plt.figure(figsize=(12, 6))
plt.title("Distribution of month")
plt.hist(dt["month"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("month")
plt.ylabel("Frequency")

plt.xticks(rotation=90)
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột weekday
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
plt.figure(figsize=(12, 6))
plt.title("Distribution of weekday")
plt.hist(dt["weekday"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("weekday")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột season
codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
dt["season"] = dt["season"].map(codes)
plt.figure(figsize=(12, 6))
plt.title("Distribution of season")
plt.hist(dt["season"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("season")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột weather
codes = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
dt["weather"] = dt["weather"].map(codes)
plt.figure(figsize=(12, 6))
plt.title("Distribution of weather")
plt.hist(dt["weather"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("weather")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột workingday
codes = {1: "working day", 0: "not workingday"}
dt["workingday"] = dt["workingday"].map(codes)
plt.figure(figsize=(12, 6))
plt.title("Distribution of workingday")
plt.hist(dt["workingday"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("workingday")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột workingday
codes = {1: "holiday", 0: "not holiday"}
dt["holiday"] = dt["holiday"].map(codes)
plt.figure(figsize=(12, 6))
plt.title("Distribution of holiday")
plt.hist(dt["holiday"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("holiday")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()


# vẽ biểu đồ phân phối của cột temp
plt.figure(figsize=(12, 6))
plt.title("Distribution of temp")
plt.hist(dt["temp"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("temp")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột atemp
plt.figure(figsize=(12, 6))
plt.title("Distribution of atemp")
plt.hist(dt["atemp"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("atemp")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()


# vẽ biểu đồ phân phối của cột humidity
plt.figure(figsize=(12, 6))
plt.title("Distribution of humidity")
plt.hist(dt["humidity"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("humidity")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()


# vẽ biểu đồ phân phối của cột windspeed
plt.figure(figsize=(12, 6))
plt.title("Distribution of windspeed")
plt.hist(dt["windspeed"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("windspeed")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()

# vẽ biểu đồ phân phối của cột count
plt.figure(figsize=(12, 6))
plt.title("Distribution of count")
plt.hist(dt["count"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("count")
plt.ylabel("Frequency")
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.2, top=0.5, right=0.5, wspace=0.2, hspace=0.2)
plt.show()
