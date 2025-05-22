# Demand Forecasting Model

This project is a real-world demand forecasting pipeline designed to analyze retail sales patterns and build predictive models for better decision-making in supply chain and inventory planning.

## 📊 Objective
To predict future product demand using machine learning and time series forecasting methods. The insights are intended to support inventory management, pricing strategies, and promotional planning.

## 📁 Dataset
The dataset includes 2,400 entries across multiple products with features such as:
- Product_ID
- Product_Name
- Category
- Price
- Promo_Flag (whether a promotion is running)
- Month (as datetime)
- Units_Sold (target variable)
- Year, Is_Seasonal, Holiday_Flag

## 🧪 Key Steps

### 1. Data Exploration & Preprocessing
- Time-based trends, promo effects, and category distributions visualized.
- Conversion of categorical features and datetime formatting.

### 2. Machine Learning with XGBoost
- Trained a regression model using encoded features.
- Performance Metrics:
  - MAE: 8.47
  - RMSE: 10.56
  - R² Score: 0.83

### 3. Time Series Forecasting with Prophet
- Focused on a specific product (`P1000`)
- Captured trend and seasonality for 6-month forecasts.

### 4. Visualization
- Monthly sales trend
- Impact of promotion
- Feature importance
- Actual vs. predicted sales (XGBoost)
- Forecast charts (Prophet)

## 🛠️ Tools & Libraries
- Python (pandas, matplotlib, seaborn)
- XGBoost
- Prophet (by Meta)
- scikit-learn

## 📈 Use Cases
- Retail demand planning
- Promotional effectiveness analysis
- Inventory and supply chain optimization

## 📎 File Structure
- `demand_forecasting.py`: Full pipeline code
- Dataset: `demand_forecasting_dataset.csv`
- Screenshots and plots (optional)

## 👤 Author
Osman Karamujić

---

Feel free to fork this repo, raise issues, or contribute improvements!
