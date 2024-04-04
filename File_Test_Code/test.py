import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno

# Ignore  the warnings
import warnings

warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

# plt.style.use('fivethirtyeight')
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

mydata = pd.read_csv("./train.csv", parse_dates=True, index_col="datetime")
testdata = pd.read_csv("./test.csv", parse_dates=True, index_col="datetime")
print("Shape of data: ", mydata.shape)
print(mydata.head(3))
print(mydata.info())
print(mydata.describe())
print(mydata.index[[0, -1]])  # Range of time stamp
print(
    "Casual + Registered = Count? ",
    ~(mydata.casual + mydata.registered - mydata["count"]).any(),
)
# Converting into categorical data
category_list = ["season", "holiday", "workingday", "weather"]
for var in category_list:
    mydata[var] = mydata[var].astype("category")
    testdata[var] = testdata[var].astype("category")
# Mapping numbers to understandable text
season_dict = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
weather_dict = {
    1: "Clear",
    2: "Misty+Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Snow/Rain",
}
mydata["season"] = mydata["season"].map(season_dict)
mydata["weather"] = mydata["weather"].map(weather_dict)

testdata["season"] = testdata["season"].map(season_dict)
testdata["weather"] = testdata["weather"].map(weather_dict)

print(mydata.head(n=3))

# Average values across each of the categorical columns
fig = plt.figure(figsize=(15, 12))
axes = fig.add_subplot(2, 2, 1)
group_weather = pd.DataFrame(mydata.groupby(["weather"])["count"].mean()).reset_index()
sns.barplot(data=group_weather, x="weather", y="count", ax=axes)
axes.set(xlabel="Weather", ylabel="Count", title="Average bike rentals across Weather")

axes = fig.add_subplot(2, 2, 2)
group_season = pd.DataFrame(mydata.groupby(["season"])["count"].mean()).reset_index()
sns.barplot(data=group_season, x="season", y="count", ax=axes)
axes.set(xlabel="Season", ylabel="Count", title="Average bike rentals across Seasons")

axes = fig.add_subplot(2, 2, 3)
group_workingday = pd.DataFrame(
    mydata.groupby(["workingday"])["count"].mean()
).reset_index()
sns.barplot(data=group_workingday, x="workingday", y="count", ax=axes)
axes.set(
    xlabel="Working Day",
    ylabel="Count",
    title="Average bike rentals across Working Day",
)

axes = fig.add_subplot(2, 2, 4)
group_season = pd.DataFrame(mydata.groupby(["holiday"])["count"].mean()).reset_index()
sns.barplot(data=group_season, x="holiday", y="count", ax=axes)
axes.set(xlabel="Holiday", ylabel="Count", title="Average bike rentals across Holiday")
plt.show()

# Seaborn boxplots to get an idea of the distribution/outliers
f, axes = plt.subplots(2, 2, figsize=(15, 12))
hue_order = ["Clear", "Heavy Snow/Rain", "Light Snow/Rain", "Misty+Cloudy"]
sns.boxplot(data=mydata, y="count", x="weather", ax=axes[0][0], order=hue_order)
sns.boxplot(data=mydata, y="count", x="workingday", ax=axes[0][1])
hue_order = ["Fall", "Spring", "Summer", "Winter"]
sns.boxplot(data=mydata, y="count", x="season", ax=axes[1][0], order=hue_order)
sns.boxplot(data=mydata, y="count", x="holiday", ax=axes[1][1])

plt.show()


# Splitting data into working-day and non-working day
mydata_w = mydata[mydata.workingday == 1]
mydata_nw = mydata[mydata.workingday == 0]

bin_size = 4
mydata_w["temp_round"] = mydata_w["temp"] // bin_size
mydata_nw["temp_round"] = mydata_nw["temp"] // bin_size

mean_count_vs_temp_w = mydata_w.groupby("temp_round")["count"].mean()
mean_count_vs_temp_nw = mydata_nw.groupby("temp_round")["count"].mean()
idx_w, idx_nw = range(len(mean_count_vs_temp_w)), range(len(mean_count_vs_temp_nw))
labels_w = [
    str(bin_size * i) + " to " + str(bin_size * (i + 1))
    for i in range(len(mean_count_vs_temp_w))
]
labels_nw = [
    str(bin_size * i) + " to " + str(bin_size * (i + 1))
    for i in range(len(mean_count_vs_temp_nw))
]

fig = plt.figure(figsize=(18, 6))
axes = fig.add_subplot(1, 2, 1)
plt.bar(x=idx_w, height=mean_count_vs_temp_w)
plt.xticks(idx_w, labels_w, rotation=90)
plt.xlabel("temp bins")
plt.ylabel("Average Count")
plt.title("Working Days: Average Count given across temperature range")

axes = fig.add_subplot(1, 2, 2)
plt.bar(x=idx_nw, height=mean_count_vs_temp_nw)
plt.xticks(idx_nw, labels_nw, rotation=90)
plt.xlabel("temp bins")
plt.ylabel("Average Count")
plt.title("Non-Working Days: Average Count given across temperature range")

plt.show()

msno.matrix(mydata)


# Splitting datetime object into month, date, hour and day category columns
mydata["month"] = [x.month for x in mydata.index]
mydata["date"] = [x.day for x in mydata.index]
mydata["hour"] = [x.hour for x in mydata.index]
mydata["day"] = [x.weekday() for x in mydata.index]

testdata["month"] = [x.month for x in testdata.index]
testdata["date"] = [x.day for x in testdata.index]
testdata["hour"] = [x.hour for x in testdata.index]
testdata["day"] = [x.weekday() for x in testdata.index]

category_list = ["month", "date", "hour", "day"]
for var in category_list:
    mydata[var] = mydata[var].astype("category")
    testdata[var] = testdata[var].astype("category")

# Mapping 0 to 6 day indices to Monday to Saturday
day_dict = {
    0: "Monday",
    1: "Teusday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
mydata["day"] = mydata["day"].map(day_dict)
testdata["day"] = testdata["day"].map(day_dict)

mydata.head(n=3)

# seaborn boxplots across hours
f, axes = plt.subplots(1, 1, figsize=(15, 6))
sns.boxplot(data=mydata, y="count", x="hour", hue="workingday", ax=axes)
handles, _ = axes.get_legend_handles_labels()
axes.legend(handles, ["Not a Working Day", "Working Day"])
axes.set(title="Hourly Count based on Working day or not")

plt.show()


# Plots of average count across hour in a day for various categories

f, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 18))
group_work_hour = pd.DataFrame(
    mydata.groupby(["workingday", "hour"])["count"].mean()
).reset_index()
sns.pointplot(
    data=group_work_hour, x="hour", y="count", hue="workingday", ax=axes[0], legend=True
)
handles, _ = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ["Not a Working Day", "Working Day"])
axes[0].set(
    xlabel="Hour in the day",
    ylabel="Count",
    title="Average Bike Rentals by the day if Working day or Not",
)

hue_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
group_day_hour = pd.DataFrame(
    mydata.groupby(["day", "hour"])["count"].mean()
).reset_index()
sns.pointplot(
    data=group_day_hour, x="hour", y="count", hue="day", ax=axes[1], hue_order=hue_order
)
axes[1].set(
    xlabel="Hour in the day",
    ylabel="Count",
    title="Average Bike Rentals by the day across Weekdays",
)

df_melt = pd.melt(
    frame=mydata,
    id_vars="hour",
    value_vars=["casual", "registered"],
    value_name="count",
    var_name="casual_or_registered",
)
group_casual_hour = pd.DataFrame(
    df_melt.groupby(["hour", "casual_or_registered"])["count"].mean()
).reset_index()
sns.pointplot(
    data=group_casual_hour, x="hour", y="count", hue="casual_or_registered", ax=axes[2]
)
axes[2].set(
    xlabel="Hour in the day",
    ylabel="Count",
    title="Average Bike Rentals by the day across Casual/Registered Users",
)

plt.show()


# Average Monthly Count Distribution plot
f, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
group_month = pd.DataFrame(
    mydata.groupby(["month", "workingday"])["count"].mean()
).reset_index()
sns.barplot(data=group_month, x="month", y="count", hue="workingday", ax=axes)
axes.set(xlabel="Month", ylabel="Count", title="Average bike rentals per Month")
handles, _ = axes.get_legend_handles_labels()
axes.legend(handles, ["Not a Working Day", "Working Day"])
plt.show()


mydata_w = mydata[mydata.workingday == 1]
mydata_nw = mydata[mydata.workingday == 0]

fig = plt.figure(figsize=(18, 8))
# Working Day
axes = fig.add_subplot(1, 2, 1)
f = axes.scatter(mydata_w.hour, mydata_w["count"], c=mydata_w.temp, cmap="RdBu")
axes.set(
    xticks=range(24),
    xlabel="Hours in day",
    ylabel="Count",
    title="Working Day: Count vs. Day Hour with Temperature Gradient",
)
cbar = plt.colorbar(f)
cbar.set_label("Temperature in degree C")

# Non Working Day
axes = fig.add_subplot(1, 2, 2)
f = axes.scatter(mydata_nw.hour, mydata_nw["count"], c=mydata_nw.temp, cmap="RdBu")
axes.set(
    xticks=range(24),
    xlabel="Hours in day",
    ylabel="Count",
    title="Non Working Day: Count vs. Day Hour with Temperature Gradient",
)
cbar = plt.colorbar(f)
cbar.set_label("Temperature in degree C")

plt.show()

heavy_weather_data = mydata.loc[mydata["weather"] == "Heavy Snow/Rain", :]
print(heavy_weather_data.index)
mydata["2012-01-09 08:00":"2012-01-09 20:00"]
