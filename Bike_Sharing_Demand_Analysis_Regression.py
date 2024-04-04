import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.options.display.max_columns = 999

# Đọc dữ liệu từ file "train.csv" và lưu vào DataFrame df
df = pd.read_csv("train.csv")

# Tạo cột year, month, day, hour từ cột datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# In ra 5 dòng đầu tiên của DataFrame df
print(df.head())

# Thống kê thông tin mô tả của DataFrame df
print(df.describe())

# Đếm số giá trị duy nhất trong mỗi cột của DataFrame df
print(df.apply(lambda x: len(x.unique())))

# Kiểm tra xem có giá trị null nào trong DataFrame df không
print(df.isnull().sum())

# Xóa cột "datetime" khỏi DataFrame df
df = df.drop(columns=["datetime"])

# Chuyển đổi các cột kiểu int thành kiểu category
cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]
for col in cols:
    df[col] = df[col].astype("category")
print(df.info())

# Phân tích dữ liệu

# Vẽ biểu đồ điểm (pointplot) thể hiện số lượng xe đạp theo giờ trong tuần
fig, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="count", hue="weekday", ax=ax)
ax.set(title="Số lượng xe đạp theo giờ trong tuần")

# Vẽ biểu đồ điểm (pointplot) thể hiện số lượng xe đạp của người dùng không đăng ký theo giờ trong tuần
fig, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="casual", hue="weekday", ax=ax)
ax.set(title="Số lượng xe đạp của người dùng không đăng ký theo giờ trong tuần")

# Vẽ biểu đồ điểm (pointplot) thể hiện số lượng xe đạp của người dùng đã đăng ký theo giờ trong tuần
fig, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="registered", hue="weekday", ax=ax)
ax.set(title="Số lượng xe đạp của người dùng đã đăng ký theo giờ trong tuần")

# Vẽ biểu đồ điểm (pointplot) thể hiện số lượng xe đạp theo thời tiết
fig, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="count", hue="weather", ax=ax)
ax.set(title="Số lượng xe đạp theo thời tiết")

# Vẽ biểu đồ điểm (pointplot) thể hiện số lượng xe đạp theo mùa
fig, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="count", hue="season", ax=ax)
ax.set(title="Số lượng xe đạp theo mùa")

# Vẽ biểu đồ cột (barplot) thể hiện số lượng xe đạp theo từng tháng
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(data=df, x="month", y="count", ax=ax)
ax.set(title="Số lượng xe đạp theo từng tháng")

# Vẽ biểu đồ cột (barplot) thể hiện số lượng xe đạp theo từng ngày trong tuần
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(data=df, x="weekday", y="count", ax=ax)
ax.set(title="Số lượng xe đạp theo từng ngày trong tuần")

# Vẽ biểu đồ scatter (regplot) thể hiện mối quan hệ giữa nhiệt độ và số lượng người sử dụng xe đạp
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
sns.regplot(x=df["temp"], y=df["count"], ax=ax1)
ax1.set(title="Mối quan hệ giữa nhiệt độ và số lượng người sử dụng xe đạp")

# Vẽ biểu đồ scatter (regplot) thể hiện mối quan hệ giữa độ ẩm và số lượng người sử dụng xe đạp
sns.regplot(x=df["humidity"], y=df["count"], ax=ax2)
ax2.set(title="Mối quan hệ giữa độ ẩm và số lượng người sử dụng xe đạp")

# Vẽ biểu đồ phân phối (distplot) của số lượng người sử dụng xe đạp
# và biểu đồ qqplot thể hiện sự phân phối của dữ liệu so với phân phối lý thuyết

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
sns.distplot(df["count"], ax=ax1)
ax1.set(title="Phân phối của số lượng người sử dụng xe đạp")
qqplot(df["count"], ax=ax2, line="s")
ax2.set(title="Các điểm lý thuyết")

# Áp dụng logarithm cho cột "count" để giảm độ biến thiên của dữ liệu
df["count"] = np.log(df["count"])

# Vẽ biểu đồ phân phối (distplot) của số lượng người sử dụng xe đạp sau khi áp dụng logarithm
# và biểu đồ qqplot thể hiện sự phân phối của dữ liệu so với phân phối lý thuyết
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
sns.distplot(df["count"], ax=ax1)
ax1.set(title="Phân phối của số lượng người sử dụng xe đạp sau khi áp dụng logarithm")
qqplot(df["count"], ax=ax2, line="s")
ax2.set(title="Các điểm lý thuyết")

# Tính ma trận tương quan giữa các cột trong DataFrame df
corr = df.corr()

# Vẽ biểu đồ heatmap thể hiện ma trận tương quan
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, annot_kws={"size": 15})

# Áp dụng one-hot encoding cho các cột "season", "month", "hour", "holiday", "weekday", "workingday", "weather"
def one_hot_encoding(data, column):
    data = pd.concat(
        [data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1
    )
    data = data.drop([column], axis=1)
    return data

cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]
df_oh = df
for col in cols:
    df_oh = one_hot_encoding(df_oh, col)
df_oh.head()

# Tạo X và y từ DataFrame df_oh
X = df_oh.drop(columns=["atemp", "windspeed", "casual", "registered", "count"], axis=1)
y = df_oh["count"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest Regression và huấn luyện mô hình trên tập huấn luyện
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(x_test)

# Tính và in ra giá trị MSE và RMSE
print("MSE = ", (mean_squared_error(y_test, y_pred)))
print("RMSE = ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Hiển thị các biểu đồ đã vẽ
# plt.show()
