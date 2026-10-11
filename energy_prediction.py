from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

df_train = pd.read_csv("../f1laptimesregression/ashrae3-train.csv")
df_weather = pd.read_csv("../f1laptimesregression/ashrae3-weather_train.csv")
df_building = pd.read_csv("../f1laptimesregression/ashrae3-building_metadata.csv")

df1 = df_train.merge(
    df_building,
    on=["building_id"],
    how="left"
)

df = df1.merge(
    df_weather,
    on=["site_id", "timestamp"],
    how="left"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday

df = df.drop(columns=["timestamp"])

df = df.dropna()

x = df.drop(columns=["building_id", "site_id", "meter_reading"])
x = pd.get_dummies(x, columns=["primary_use"])

y = df["meter_reading"]

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=.20, random_state=0)

forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest.fit(x_train, y_train)
y_pred1 = forest.predict(x_test)

linear = LinearRegression()
linear.fit(x_train, y_train)
y_pred2 = linear.predict(x_test)

models = {
    "Linear Regression": y_pred2,
    "Random Forest": y_pred1
}

for name, predictions in models.items():
    print(f"\n{name}")
    print(f"R²: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, predictions):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(y_test, y_pred2, alpha=0.6)
ax[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
ax[0].set_title("Linear Regression")
ax[0].set_xlabel("Actual")
ax[0].set_ylabel("Predicted")

ax[1].scatter(y_test, y_pred1, alpha=0.6)
ax[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
ax[1].set_title("Random Forest")
ax[1].set_xlabel("Actual")
ax[1].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=300)
plt.show()
