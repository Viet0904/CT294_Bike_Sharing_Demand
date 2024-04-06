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
# plt.xticks(rotation=0)  # Rotate x-axis labels
# plt.show()

# Tôi cần biễu diễn sự phân phối dữ liệu của các thuộc tính
# trong tập dữ liệu
# Lặp qua từng cột (trừ cột datetime)
for column in dt.columns[0:]:
    # Vẽ biểu đồ phân phối của cột hiện tại
    plt.figure(figsize=(8, 6))
    plt.hist(dt[column], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of ' + column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
