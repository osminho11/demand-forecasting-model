import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

df = pd.read_csv("demand_forecasting_dataset.csv")
print("✅ Dataset loaded successfully.")
print("Shape:", df.shape)
print("\n=== Sample rows ===")
print(df.head())
print("\n=== Data Types and Missing Values ===")
print(df.info())
print("\n=== Null values per column ===")
print(df.isnull().sum())
print("\n=== Summary Statistics ===")
print(df.describe())

df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values(by=['Product_ID', 'Month'])

sample_product = df[df['Product_ID'] == 'P1000']
plt.figure(figsize=(10, 5))
plt.plot(sample_product['Month'], sample_product['Units_Sold'], marker='o')
plt.title("Monthly Units Sold for Product P1000")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.grid(True)
plt.tight_layout()
plt.show()

promo_group = df.groupby('Promo_Flag')['Units_Sold'].mean()
print("\n=== Average Units Sold by Promo Flag ===")
print(promo_group)

sns.barplot(x=promo_group.index, y=promo_group.values)
plt.xticks([0, 1], ['No Promo', 'Promo'])
plt.title("Average Units Sold - Promo vs No Promo")
plt.ylabel("Average Units Sold")
plt.xlabel("Promotion Flag")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=True)
df_encoded = df_encoded.drop(columns=['Product_Name', 'Month', 'Product_ID'])

X = df_encoded.drop('Units_Sold', axis=1)
y = df_encoded['Units_Sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n=== XGBoost Model Performance ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

product_df = df[df['Product_ID'] == 'P1000'].copy()
product_df.rename(columns={"Month": "ds", "Units_Sold": "y"}, inplace=True)
product_df = product_df[["ds", "y"]]

model = Prophet()
model.fit(product_df)
future = model.make_future_dataframe(periods=6, freq='MS')
forecast = model.predict(future)

fig = model.plot(forecast)
plt.title("Prophet Forecast for Product P1000")
plt.tight_layout()
plt.show()

fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title("XGBoost: Actual vs Predicted Units Sold")
plt.grid(True)
plt.tight_layout()
plt.show()

importances = xgb_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
