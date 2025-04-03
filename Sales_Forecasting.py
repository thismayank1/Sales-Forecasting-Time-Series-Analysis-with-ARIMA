import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
file_path = "Chocolate Sales.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Clean 'Amount' column by removing '$' and converting to float
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)

# Aggregate sales by date
df_grouped = df.groupby('Date')['Amount'].sum().reset_index()

# Plot sales trends
df_grouped.set_index('Date').plot(figsize=(12, 6), title='Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.show()

# Train ARIMA model
model = ARIMA(df_grouped['Amount'], order=(5,1,0))  # (p,d,q) values can be tuned
model_fit = model.fit()

# Forecast future sales
forecast_steps = 30  # Predict next 30 days
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df_grouped['Date'].max(), periods=forecast_steps+1, freq='D')[1:]
forecast_df = pd.DataFrame({'Date': forecast_index, 'Predicted Sales': forecast})

# Plot forecast using Plotly
fig = px.line(df_grouped, x='Date', y='Amount', title='Sales Forecast')
fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted Sales'], mode='lines', name='Forecast')
fig.show()

# Save cleaned dataset and forecast results
df_grouped.to_csv("Cleaned_Chocolate_Sales.csv", index=False)
forecast_df.to_csv("Sales_Forecast.csv", index=False)
