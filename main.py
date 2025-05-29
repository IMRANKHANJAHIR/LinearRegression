

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\30 day intesnhip online\Housing.csv")
print("First 5 rows of the dataset:")
print(df.head())

print("\nChecking for missing values:")
print(df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

print("\nModel Coefficients:")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

if "area" in X.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test["area"], y_test, color='blue', label='Actual')
    plt.scatter(X_test["area"], y_pred, color='red', label='Predicted')
    plt.plot(X_test["area"], y_pred, color='black', linewidth=1)
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Linear Regression: Area vs Price')
    plt.legend()
    plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()
