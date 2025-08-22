

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


print("=== 1. Data Preprocessing ===")
data = pd.read_csv("tip.csv")
print("\nOriginal Data:\n", data.head())


imputer = SimpleImputer(strategy='mean')
numeric_data = data.select_dtypes(include=[np.number])
numeric_cleaned = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

non_numeric_data = data.select_dtypes(exclude=[np.number]).reset_index(drop=True)
data_cleaned = pd.concat([numeric_cleaned, non_numeric_data], axis=1)
print("\nData after SimpleImputer:\n", data_cleaned.head())

X = data_cleaned[['total_bill', 'size']]
y = data_cleaned['tip']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Features (first 5 rows):\n", X_scaled[:5])

data_dropped = data.dropna()
print("\nData after dropping missing rows:\n", data_dropped.head())

Sales)

print("\n=== 2. Linear Regression (Ice Cream Sales) ===")
icecream_data = pd.DataFrame({
    "Temp": [20, 25, 30, 35, 40],
    "IceCreamSales": [13, 21, 25, 35, 38]
})

X_ice = icecream_data[["Temp"]]
y_ice = icecream_data["IceCreamSales"]

ice_model = LinearRegression()
ice_model.fit(X_ice, y_ice)

print("Intercept:", ice_model.intercept_)
print("Coefficient:", ice_model.coef_[0])

pred_temp = 28
pred_sales = ice_model.predict(pd.DataFrame({"Temp": [pred_temp]}))
print(f"Predicted sales at {pred_temp}°C: {pred_sales[0]:.2f} litres")

plt.figure(figsize=(6, 4))
plt.scatter(X_ice, y_ice, color='blue', label='Actual Sales')
plt.plot(X_ice, ice_model.predict(X_ice), color='red', label='Regression Line')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (litres)')
plt.title('Ice Cream Sales vs Temperature')
plt.legend()
plt.grid(True)
plt.show()



print("\n=== 3. Cross-Validation (Coffee Sales) ===")
coffee_data = pd.DataFrame({
    "Temp": [20, 25, 30, 35, 40],
    "CoffeeSales": [45, 37, 28, 22, 18]
})

X_coffee = coffee_data[["Temp"]]
y_coffee = coffee_data["CoffeeSales"]

coffee_model = LinearRegression()
scores = cross_val_score(coffee_model, X_coffee, y_coffee, cv=2, scoring='r2')
print("Cross-validation scores:", scores)
print("Average R² score:", np.mean(scores))

coffee_model.fit(X_coffee, y_coffee)
plt.figure(figsize=(6, 4))
plt.scatter(X_coffee, y_coffee, color='brown', label='Actual Sales')
plt.plot(X_coffee, coffee_model.predict(X_coffee), color='green', label='Regression Line')
plt.xlabel('Temperature (°C)')
plt.ylabel('Coffee Sales (litres)')
plt.title('Coffee Sales vs Temperature')
plt.legend()
plt.grid(True)
plt.show()












