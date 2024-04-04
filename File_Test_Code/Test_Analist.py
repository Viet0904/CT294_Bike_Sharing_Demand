import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data from csv file
bike_sharing = pd.read_csv("train.csv")
# Tạo cột year, month, day, hour từ cột datetime
bike_sharing["datetime"] = pd.to_datetime(bike_sharing["datetime"])
bike_sharing["year"] = bike_sharing["datetime"].dt.year
bike_sharing["month"] = bike_sharing["datetime"].dt.month
bike_sharing["day"] = bike_sharing["datetime"].dt.day
bike_sharing["hour"] = bike_sharing["datetime"].dt.hour


# sns.pairplot(
#     bike_sharing[
#         ["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]
#     ]
# )
# plt.show()

# Chuẩn bị dữ liệu
bike_sharing.drop(columns=["datetime", "casual", "registered"], axis=1, inplace=True)
print(bike_sharing.head())

# Biến giả
season_type = pd.get_dummies(bike_sharing["season"], drop_first=True)
season_type.rename(
    columns={2: "season_summer", 3: "season_fall", 4: "season_winter"}, inplace=True
)
print(season_type.head())

weather_type = pd.get_dummies(bike_sharing["weather"], drop_first=True)
weather_type.rename(
    columns={2: "weather_mist_cloud", 3: "weather_light_snow_rain"}, inplace=True
)

print(weather_type.head())

# Concatenating new dummy variables to the main dataframe

bike_sharing = pd.concat([bike_sharing, season_type, weather_type], axis=1)

# Dropping columns season & weathersit since we have already created dummies for them

bike_sharing.drop(columns=["season", "weather"], axis=1, inplace=True)

# Analysing dataframe after dropping columns

print(bike_sharing.info())

# Tạo các biến dẫn xuất cho tháng biến phân loại

# Creating year_quarter derived columns from month columns.
# Note that last quarter has not been created since we need only 3 columns to define the four quarters.

bike_sharing["Quarter_JanFebMar"] = bike_sharing["month"].apply(
    lambda x: 1 if x <= 3 else 0
)
bike_sharing["Quarter_AprMayJun"] = bike_sharing["month"].apply(
    lambda x: 1 if 4 <= x <= 6 else 0
)
bike_sharing["Quarter_JulAugSep"] = bike_sharing["month"].apply(
    lambda x: 1 if 7 <= x <= 9 else 0
)

# Dropping column mnth since we have already created dummies.

bike_sharing.drop(columns=["month"], axis=1, inplace=True)
bike_sharing["day"] = bike_sharing["day"].apply(lambda x: 0 if 1 <= x <= 5 else 1)
bike_sharing.drop(columns=["day"], axis=1, inplace=True)
bike_sharing.drop(columns=["workingday"], axis=1, inplace=True)
print(bike_sharing.head())
# Analysing dataframe after dropping columns day & workingday

print(bike_sharing.info())


# Plotting correlation heatmap to analyze the linearity between the variables in the dataframe

plt.figure(figsize=(16, 10))
sns.heatmap(bike_sharing.corr(), annot=True, cmap="Greens")
plt.show()


# Plotting correlation heatmap to analyze the linearity between the variables in the dataframe

plt.figure(figsize=(16, 10))
sns.heatmap(bike_sharing.corr(), annot=True, cmap="Greens")
plt.show()


# Dropping column temp since it is very highly collinear with the column atemp.
# Further,the column atemp is more appropriate for modelling compared to column temp from human perspective.

bike_sharing.drop(columns=["temp"], axis=1, inplace=True)
print(bike_sharing.head())


# Chia dữ liệu thành các bộ đào tạo và kiểm tra


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
bike_sharing_train, bike_sharing_test = train_test_split(
    bike_sharing, train_size=0.8, test_size=0.2, random_state=100
)


# importing library
from sklearn.preprocessing import MinMaxScaler

# assigning variable to scaler
scaler = MinMaxScaler()
# Applying scaler to all the columns except the derived and 'dummy' variables that are already in 0 & 1.

numeric_var = ["atemp", "humidity", "windspeed", "count"]
bike_sharing_train[numeric_var] = scaler.fit_transform(bike_sharing_train[numeric_var])

# Analysing the train dataframe after scaling
print(bike_sharing_train.head())

y_train = bike_sharing_train.pop("count")
X_train = bike_sharing_train

print(y_train.head())
print(X_train.head())

# Xây dựng mô hình tuyến tính


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Running RFE with the output number of the variable equal to 12
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 12)  # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[columns_rfe]
X_train_rfe


# Phân tích còn lại dữ liệu đào tạo

# using the final model lr5 on train data to predict y_train_cnt values
y_train_cnt = lr5.predict(X_train_lr5)
# Plotting the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins=20)
fig.suptitle("Error Terms", fontsize=20)
plt.xlabel("Errors", fontsize=18)


plt.scatter(y_train, (y_train - y_train_cnt))
plt.show()

# Applying the scaling on the test sets

numeric_vars = ["atemp", "humidity", "windspeed", "count"]
bike_sharing_test[numeric_vars] = scaler.transform(bike_sharing_test[numeric_vars])

bike_sharing_test.describe()


y_test = bike_sharing_test.pop("count")
X_test = bike_sharing_test

# Adding constant variable to test dataframe
X_test_lr5 = sm.add_constant(X_test)

# Updating X_test_lr5 dataframe by dropping the variables as analyzed from the above models

X_test_lr5 = X_test_lr5.drop(
    [
        "atemp",
        "hum",
        "season_fall",
        "Quarter_AprMayJun",
        "weekend",
        "Quarter_JanFebMar",
    ],
    axis=1,
)

# Making predictions using the fifth model

y_pred = lr5.predict(X_test_lr5)


# Đánh giá mô hình


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle("y_test vs y_pred", fontsize=20)
plt.xlabel("y_test", fontsize=18)

plt.ylabel("y_pred", fontsize=16)


# importing library and checking mean squared error
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean_Squared_Error :", mse)

# importing library and checking R2

from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))