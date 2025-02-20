# TIME SERIES ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

pd.options.display.max_columns = None


data = pd.read_excel(
    r"C:\Users\\Classic Models Sales.xlsx")

# Removing Columns i wont be using for analysis and setting date as an index.
# Important to check if date has datatype as datetime
data['orderdate'] = pd.to_datetime(data['orderdate'])
data.set_index('orderdate', inplace=True)
data = data.drop(['ordernumber', 'productName', 'productLine', 'customerName',
                  'customer_city', 'customer_country', 'office_city', 'office_country',
                  'employee_first_name', 'employee_last_name', 'buyPrice', 'priceEach',
                  'QuantityOrdered', 'cost_of_sales'], axis=1, inplace=False)

data.info()
# Basic Intial View
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(data.index, data['sales_value'])
plt.title('Classic Cars Sales (2003-2005)')
plt.xlabel('Date')
plt.ylabel('Sales Amount')

# Resampling data to Monthly Sales Date
monthly_sales = data['sales_value'].resample('ME').sum()

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(monthly_sales)
plt.title('Monthly Classic Cars Sales (2003-2005)')
plt.xlabel('Date')
plt.ylabel('Sales Amount')

# Checking for stationarity:
# ADF Test
adf_test = adfuller(monthly_sales)
if adf_test[1] <= 0.05:
    print('\nSeries is Stationary')
else:
    print('\nSeries is not Stationary requires Differencing')


# Checking for Seasonality and AR/MA:
# Using ACF/PACF

monthly_sales_diff = monthly_sales.diff(12).dropna()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plot_acf(monthly_sales, lags=12, ax=axes[0, 0])
axes[0, 0].set_title("ACF - Before Seasonal Differencing")
plot_pacf(monthly_sales, lags=12, ax=axes[0, 1], method='ywm')
axes[0, 1].set_title("PACF - Before Seasonal Differencing")
plt.subplot(2, 2, 1)
plot_acf(monthly_sales_diff, lags=7, ax=axes[1, 0])
axes[1, 0].set_title("ACF - After Seasonal Differencing")
plt.subplot(2, 2, 2)
plot_pacf(monthly_sales_diff, lags=7, ax=axes[1, 1], method='ywm')
axes[1, 1].set_title("PACF - After Seasonal Differencing")
plt.tight_layout()
plt.show()


# Decompose Data:
decomposition_additive = seasonal_decompose(monthly_sales,
                                            model='additive')
decomposition_multiplicative = seasonal_decompose(
    monthly_sales, model='multiplicative')

plt.rcParams.update({'figure.figsize': (16, 12)})
decomposition_additive.plot().suptitle('Additive Decomposition', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
decomposition_multiplicative.plot().suptitle(
    'Multiplicative Decomposition', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Model Selection and Fitting:
# SARIMA model

sarima_model = SARIMAX(monthly_sales, order=(
    1, 0, 1), seasonal_order=(1, 1, 1, 12))

sarima_result = sarima_model.fit()
sarima_result.summary()
sarima_result.plot_diagnostics(figsize=(12, 6))


# Forecasting:

forecast = sarima_result.get_forecast(steps=12)
forecast_index = pd.date_range(
    start=monthly_sales.index[-1], periods=12, freq='ME')
forecast_series = forecast.predicted_mean
forecast_series.index = forecast_index

plt.figure(figsize=(16, 12))
plt.plot(monthly_sales, label='Historical Sales')
plt.plot(forecast_series, label='Forecasted Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Sales Value')
plt.title('Sales Forecast using ARIMA')
plt.legend()
plt.show()

# Model Evaluation:
forecast = sarima_result.get_forecast(steps=12)
forecast_series = forecast.predicted_mean
actual_sales = monthly_sales.iloc[-12:]
mse = mean_squared_error(monthly_sales[-12:], forecast_series)
mae = mean_absolute_error(monthly_sales[-12:], forecast_series)
rmse = root_mean_squared_error(monthly_sales[-12:], forecast_series)
