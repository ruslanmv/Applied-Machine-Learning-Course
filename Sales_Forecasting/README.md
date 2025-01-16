# Sales Forecasting

Welcome to the Sales Forecasting module of the Applied Machine Learning Course!

## Motivation

Accurate sales forecasting is essential for businesses to make informed decisions about inventory management, production planning, marketing campaigns, and overall business strategy. By predicting future sales, companies can optimize their operations, reduce costs, improve customer satisfaction, and ultimately increase profitability.

## Learning Objectives

By the end of this module, you will be able to:

*   Understand the fundamentals of time series analysis.
*   Perform Exploratory Data Analysis (EDA) on time series data.
*   Preprocess time series data for modeling, including handling missing values and stationarity.
*   Understand and apply ARIMA models for sales forecasting.
*   Evaluate the performance of time series models using appropriate metrics.
*   Use Python libraries like Pandas, Matplotlib, Statsmodels, and pmdarima for sales forecasting.

## Real-World Applications

*   **Retail:** Optimize inventory levels, predict demand for different products, and plan promotions.
*   **Manufacturing:** Plan production schedules, manage raw material procurement, and optimize supply chains.
*   **E-commerce:** Forecast website traffic, predict sales during peak seasons, and manage customer demand.
*   **Finance:** Predict stock prices, forecast revenue, and assess investment risks.
*   **Energy:** Forecast energy consumption and optimize energy production.

## Conceptual Overview

Sales forecasting typically involves analyzing **time series data**, which is a sequence of data points collected over time. In this module, we will focus on using **ARIMA (Autoregressive Integrated Moving Average)** models, a classic and widely used statistical method for time series forecasting.

**Key Concepts:**

*   **Time Series:** A sequence of data points collected at successive points in time, often at regular intervals.
*   **Stationarity:** A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time. Stationarity is a key assumption for many time series models, including ARIMA.
*   **Autocorrelation:** The correlation of a time series with its own past values.
*   **ARIMA Model:** A statistical model that uses past values of the time series (autoregressive part), past forecast errors (moving average part), and differencing (integrated part) to predict future values.

## Tools

*   **Python:** Our primary programming language.
*   **Pandas:** For data manipulation, analysis, and time series operations.
*   **Matplotlib:** For data visualization.
*   **Statsmodels:** For building and estimating statistical models, including ARIMA.
*   **pmdarima:** For automatic ARIMA model selection (auto\_arima).

## Dataset

We will use the **Walmart Store Sales Forecasting dataset** from Kaggle.

*   **Source:** [Kaggle - Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
*   **Description:** The dataset contains historical sales data for 45 Walmart stores, including weekly sales, holidays, fuel prices, CPI (Consumer Price Index), and unemployment rates.
*   **Features:**
    *   `Store`: Store number.
    *   `Dept`: Department number.
    *   `Date`: Week of sales.
    *   `Weekly_Sales`: Sales for the given department in the given store.
    *   `IsHoliday`: Whether the week is a special holiday week.
    *   `Temperature`: Average temperature in the region.
    *   `Fuel_Price`: Cost of fuel in the region.
    *   `CPI`: Consumer Price Index.
    *   `Unemployment`: Unemployment rate.
*   **Potential Limitations:** The dataset is from 2010-2012, so sales patterns might have changed since then. Also, external factors not included in the dataset can influence sales.

## Project Roadmap

1.  **Data Loading and Exploration:** Load the dataset, understand its structure, and perform exploratory data analysis (EDA).
2.  **Data Preprocessing:** Handle missing values, convert data types, and check for stationarity.
3.  **Feature Engineering:** Create new features if needed (e.g., lag features).
4.  **Model Selection and Training:**
    *   Manually determine ARIMA parameters or use `pmdarima`'s `auto_arima` to automatically find the best ARIMA model.
    *   Train the ARIMA model on the historical data.
5.  **Model Evaluation:** Evaluate the model's performance using appropriate metrics like MAE, MSE, RMSE.
6.  **Forecasting:** Use the trained model to predict future sales.
7.  **Interpretation and Insights:** Analyze the model's results and identify key factors influencing sales.

## Step-by-Step Instructions

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_data = pd.read_csv("train.csv")
features_data = pd.read_csv("features.csv")
stores_data = pd.read_csv("stores.csv")

# Merge the datasets
data = pd.merge(train_data, features_data, on=['Store', 'Date', 'IsHoliday'], how='left')
data = pd.merge(data, stores_data, on=['Store'], how='left')

# Convert 'Date' to datetime objects
data['Date'] = pd.to_datetime(data['Date'])

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Explore the distribution of weekly sales
plt.figure(figsize=(12, 6))
sns.histplot(data['Weekly_Sales'], bins=50)
plt.title('Distribution of Weekly Sales')
plt.show()

# Explore weekly sales over time
plt.figure(figsize=(12, 6))
data.groupby('Date')['Weekly_Sales'].sum().plot()
plt.title('Weekly Sales Over Time')
plt.ylabel('Total Weekly Sales')
plt.show()
````

**Explanation:**

  * We import the necessary libraries.
  * We load the three datasets (`train.csv`, `features.csv`, `stores.csv`) and merge them into a single DataFrame.
  * We convert the 'Date' column to the proper datetime format.
  * We examine the first few rows, data types, and check for missing values.
  * We visualize the distribution of weekly sales and plot the total weekly sales over time.

### 2\. Data Preprocessing

```python
# Handle missing values (if any) - Example: forward fill for simplicity
data.fillna(method='ffill', inplace=True)

# Check for stationarity using the Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

# Example: Test stationarity of weekly sales for a specific store and department
store_dept_sales = data[(data['Store'] == 1) & (data['Dept'] == 1)]['Weekly_Sales']
test_stationarity(store_dept_sales)
```

**Explanation:**

  * **Missing Values:** In this example, we use forward fill to impute missing values. You might need to choose a different strategy based on the nature of your data and the extent of missingness.
  * **Stationarity:** We define a function `test_stationarity` that performs the Augmented Dickey-Fuller test. This test helps determine if a time series is stationary. A low p-value (typically less than 0.05) suggests that the time series is stationary.

### 3\. Feature Engineering

```python
# Example: Create lag features for Weekly_Sales
data['Lag_1'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
data['Lag_2'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
data['Lag_3'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(3)
data['Lag_4'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
# Forward fill to deal with NAN that were created 
data.fillna(method='ffill', inplace=True)

print(data.head())
```

**Explanation:**

  * We create lag features for 'Weekly\_Sales'. Lag features are past values of the target variable, which can be useful predictors in time series models. We create lags for the past 1, 2, 3, and 4 weeks. You can experiment with different lag periods.

### 4\. Model Selection and Training

```python
# For this example, we'll focus on forecasting sales for a single store and department
store_id = 1
dept_id = 1
store_dept_data = data[(data['Store'] == store_id) & (data['Dept'] == dept_id)].copy()

# Set 'Date' as index and sort
store_dept_data.set_index('Date', inplace=True)
store_dept_data.sort_index(inplace=True)

# Split data into training and testing sets (e.g., last 4 weeks for testing)
train_data = store_dept_data[:-4]
test_data = store_dept_data[-4:]

# Use auto_arima to find the best ARIMA model
import pmdarima as pm

model = pm.auto_arima(train_data['Weekly_Sales'], 
                      seasonal=False, # Assuming no strong seasonality for simplicity
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

# Fit the best model
from statsmodels.tsa.arima.model import ARIMA

best_order = model.order  # Get the best order from auto_arima
arima_model = ARIMA(train_data['Weekly_Sales'], order=best_order)
arima_model_fit = arima_model.fit()
```

**Explanation:**

  * We select a specific store and department for this example. You can adapt the code to forecast for other stores/departments or even build a model that forecasts for all stores and departments simultaneously.
  * We set the 'Date' column as the index of the DataFrame.
  * We split the data into training and testing sets. Here, we use the last 4 weeks of data for testing.
  * **`auto_arima`:** We use the `auto_arima` function from the `pmdarima` library to automatically search for the best ARIMA model order (p, d, q).
      * `seasonal=False`: We are assuming no strong seasonality for this example. You might need to set this to `True` and specify a `seasonal_order` if your data exhibits seasonality (e.g., yearly seasonality).
      * `trace=True`: Prints the progress of the model search.
      * `stepwise=True`: Uses a stepwise algorithm to search for the best model, which is generally faster than trying all possible combinations.
  * **Fit the Model:** We create an `ARIMA` model object from `statsmodels` using the best order found by `auto_arima` and fit it to the training data.

### 5\. Model Evaluation

```python
# Forecast on the test set
forecast = arima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test_data['Weekly_Sales'], forecast)
mse = mean_squared_error(test_data['Weekly_Sales'], forecast)
rmse = np.sqrt(mse)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# Visualize the forecast vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(train_data['Weekly_Sales'], label='Train')
plt.plot(test_data['Weekly_Sales'], label='Test')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title('Weekly Sales Forecast')
plt.show()
```

**Explanation:**

  * We use the trained ARIMA model to make forecasts on the test set.
  * We evaluate the model's performance using MAE (Mean Absolute Error), MSE (Mean Squared Error), and RMSE (Root Mean Squared Error). Lower values are better for these metrics.
  * We visualize the forecast against the actual values in the test set.

### 6\. Forecasting (Future Predictions)

```python
# Forecast future sales (e.g., next 4 weeks)
future_forecast = arima_model_fit.predict(start=len(store_dept_data), end=len(store_dept_data) + 3)

# Create a date range for the future forecast
future_dates = pd.date_range(start=store_dept_data.index[-1] + pd.DateOffset(weeks=1), periods=4, freq='W')

# Create a DataFrame for the future forecast
future_forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
future_forecast_df.set_index('Date', inplace=True)

print(future_forecast_df)

# Visualize the forecast along with historical data
plt.figure(figsize=(12, 6))
plt.plot(store_dept_data['Weekly_Sales'], label='Historical')
plt.plot(future_forecast_df['Forecast'], label='Future Forecast')
plt.legend()
plt.title('Weekly Sales Forecast (Including Future)')
plt.show()
```

**Explanation:**

  * We use the trained model to forecast sales for the next 4 weeks beyond the end of the historical data.
  * We create a date range for the future forecast period.
  * We create a DataFrame to store the future forecast with the corresponding dates.
  * We visualize the historical sales data along with the future forecast.

### 7\. Interpretation and Insights

  * **Model Diagnostics:** Analyze the residuals (forecast errors) of the ARIMA model to ensure they are white noise (randomly distributed with no autocorrelation). You can use diagnostic plots provided by `arima_model_fit.plot_diagnostics()`.
  * **Feature Importance:** If you used other features besides the lagged values of `Weekly_Sales` (e.g., `IsHoliday`, `Temperature`, `CPI`, etc.), you might need to explore their impact on the forecast using techniques appropriate for the model you used to incorporate them. This could involve examining coefficients (if you used a linear model) or using feature importance scores from tree-based models.
  * **Business Insights:** Relate the forecast results to business decisions. For example, if the forecast predicts a significant increase in sales, the business might need to increase inventory or hire more staff.

## Exercises

1.  **Different Store/Department:** Apply the same analysis to a different store and department.
2.  **Seasonality:** If you suspect seasonality in the data, incorporate it into the model by setting `seasonal=True` and specifying an appropriate `seasonal_order` in `auto_arima`. You might need to use techniques like SARIMA (Seasonal ARIMA) models.
3.  **Parameter Tuning:** Experiment with different values for the `max_p`, `max_d`, `max_q` parameters in `auto_arima` to see if you can find a better model. You can also manually specify the (p, d, q) order based on ACF and PACF plots.
4.  **Exogenous Variables:** Incorporate other relevant features from the dataset (e.g., `IsHoliday`, `Temperature`, `CPI`, `Unemployment`) into the model. You can use these as exogenous variables in `auto_arima` or explore other models like SARIMAX that can handle exogenous variables.
5.  **Rolling Forecast:** Implement a rolling forecast, where you retrain the model with each new week of data and forecast the next week. This can improve accuracy but is computationally more expensive.

## Suggested Solutions (Hints)

  * **Exercise 1:** Change the `store_id` and `dept_id` variables.
  * **Exercise 2:** Use the `seasonal_order` parameter in `auto_arima`. You can use ACF and PACF plots of the differenced data to help determine the seasonal order.
  * **Exercise 3:** Refer to the `pmdarima` documentation for details on the parameters.
  * **Exercise 4:** Use the `exogenous` parameter in `auto_arima` or explore SARIMAX models in `statsmodels`.
  * **Exercise 5:** Create a loop that iterates through the test set, makes a one-step-ahead forecast, adds the actual value to the training data, and retrains the model.

## Further Resources

  * **Forecasting: Principles and Practice (Hyndman & Athanasopoulos):** [https://otexts.com/fpp3/](https://www.google.com/url?sa=E&source=gmail&q=https://otexts.com/fpp3/) (An excellent online textbook on forecasting)
  * **Statsmodels Documentation:** [https://www.statsmodels.org/stable/index.html](https://www.google.com/url?sa=E&source=gmail&q=https://www.statsmodels.org/stable/index.html)
  * **pmdarima Documentation:** [https://alkaline-ml.com/pmdarima/](https://www.google.com/url?sa=E&source=gmail&q=https://alkaline-ml.com/pmdarima/)

## Contact

For any queries or feedback related to this module, feel free to open an issue in the main repository or contact us at \[contact@ruslanmv.com].

-----

Happy Learning\!

