import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

dt = pd.read_csv("./BikeSharingDemand.csv")

dt["datetime"] = pd.to_datetime(dt["datetime"])
dt["hour"] = dt["datetime"].dt.hour
dt["month"] = dt["datetime"].dt.month
dt["weekday"] = dt["datetime"].dt.weekday
dt["year"] = dt["datetime"].dt.year

# print(dt["year"])
dt.drop("datetime", axis=1, inplace=True)
dt.drop("casual", axis=1, inplace=True)
dt.drop("registered", axis=1, inplace=True)

# Ánh xạ mã cho cột 'season'
season_codes = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}

# Ánh xạ mã cho cột 'holiday'
holiday_codes = {0: "not holiday", 1: "holiday"}

# Ánh xạ mã cho cột 'workingday'
workingday_codes = {0: "not working day", 1: "working day"}

# Ánh xạ mã cho cột 'weather'
weather_codes = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}
# Chuyển đổi giá trị trong các cột 'season', 'holiday', 'workingday', 'weather' sang các chuỗi tương ứng
dt["season"] = dt["season"].map(season_codes)
dt["holiday"] = dt["holiday"].map(holiday_codes)
dt["workingday"] = dt["workingday"].map(workingday_codes)
dt["weather"] = dt["weather"].map(weather_codes)

# Lấy các cột cần vẽ
columns_to_plot = ["season", "holiday", "workingday", "weather", "temp", "atemp"]

# Tạo một lưới biểu đồ có 2 hàng và 3 cột
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Duyệt qua từng cột và vẽ histogram
for i, column in enumerate(columns_to_plot):
    row_index = i // 3
    col_index = i % 3
    axes[row_index, col_index].hist(
        dt[column], bins=20, color="skyblue", edgecolor="black"
    )
    axes[row_index, col_index].set_title(f"Distribution of {column}")
    axes[row_index, col_index].set_xlabel(column)
    axes[row_index, col_index].set_ylabel("Frequency")

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Ánh xạ mã cho cột 'month'
month_codes = {
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

# Ánh xạ mã cho cột 'weekday'
weekday_codes = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
# Chuyển đổi giá trị trong các cột 'month' và 'weekday' sang các chuỗi tương ứng
dt["month"] = dt["month"].map(month_codes)
dt["weekday"] = dt["weekday"].map(weekday_codes)

# Lấy các cột cần vẽ
columns_to_plot = ['humidity', 'windspeed', 'hour', 'month', 'weekday', 'year', 'count']

# Tạo một lưới biểu đồ có 2 hàng và 4 cột
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))

# Duyệt qua từng cột và vẽ histogram
for i, column in enumerate(columns_to_plot):
    row_index = i // 4
    col_index = i % 4
    if column in ['month', 'weekday']:
        if column == 'month':
            labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        else:
            labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        axes[row_index, col_index].hist(dt[column], bins=20, color="skyblue", edgecolor="black")
        axes[row_index, col_index].set_title(f"Distribution of {column}")
        axes[row_index, col_index].set_xlabel(column)
        axes[row_index, col_index].set_ylabel("Frequency")
        axes[row_index, col_index].set_xticks(np.arange(len(labels)))
        axes[row_index, col_index].set_xticklabels(labels, rotation=90)  # Xoay tên chú thích trên trục X 90 độ
    elif column == 'year':
        axes[row_index, col_index].hist(dt[column], bins=20, color="skyblue", edgecolor="black")
        axes[row_index, col_index].set_title(f"Distribution of {column}")
        axes[row_index, col_index].set_xlabel(column)
        axes[row_index, col_index].set_ylabel("Frequency")
        axes[row_index, col_index].set_xticks([2011, 2012])  # Chỉ hiển thị giá trị 2011 và 2012 trên trục x
    else:
        axes[row_index, col_index].hist(dt[column], bins=20, color="skyblue", edgecolor="black")
        axes[row_index, col_index].set_title(f"Distribution of {column}")
        axes[row_index, col_index].set_xlabel(column)
        axes[row_index, col_index].set_ylabel("Frequency")

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
