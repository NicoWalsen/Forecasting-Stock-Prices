
#IMPORTING MODULES AND LIBRARIES
from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#DEFINING OUR PARAMETERS: STOCK, START DATE AND END DATE 
stock_symbol = 'BABA' #(Alibaba)
start_date = '2015-01-01'
end_date = datetime.today()

#DOWNLOADING DATA FROM YAHOO FINANCE WEBSITE AND CONVERTING IT TO A DATAFRAME IN CSV FILE
stock_data = yf.download(stock_symbol,start=start_date, end=end_date)
print(stock_data.head())

stock_data.to_csv('stock_data.csv')
stocks = pd.read_csv('stock_data.csv')
stocks.dtypes

#DROPING POSSIBLE MISSING VALUES.
stocks.dropna()

#CHANGING CLASS TYPE FOR 'DATE' FROM OBJECT TO DATATIME.
stocks['Date'] = pd.to_datetime(stocks['Date'])

#WE SELECT JUST COLUMNS "DATE" AND "CLOSE" TO START OUR ANLYSIS
stocks = stocks[['Date','Close']]
stocks
stocks.dtypes

#CHANGING THE NAME OF THE COLUMNS TO FROM "DATE" TO 'DS' AND FROM "CLOSE" TO 'Y'
stocks.columns = ['ds','y']
stocks

#PLOTING OUR HISTORICAL DATA
plt.plot(stocks['ds'],stocks['y'],label='actual', c='g')
plt.grid()
plt.show()

#TRAINING THE MODEL WITH HISTORICAL DATA: stocks
model = NeuralProphet()
model.fit(stocks)

#AFTER TRAINING OUR PREDICTOR MODEL, WE WILL FORECAST 365 DAYS INTO THE FUTURE
future = model.make_future_dataframe(stocks, periods = 365)
forecast = model.predict(future)
forecast
# WE ALSO REQUEST THE MODEL TO "PREDICT" HISTORICAL DATA
actual_prediction = model.predict(stocks)

#FINALLY, WE WILL PLOT: HISTORICAL DATA, PREDICTION OF HISTORICAL DATA AND FORECAST.
plt.plot(actual_prediction['ds'],actual_prediction['yhat1'],label="prediction_actual",c='r')
plt.plot(forecast['ds'],forecast['yhat1'],label="forecast",c='b')
plt.plot(stocks['ds'],stocks['y'],label='actual', c='g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BABA stock forecasting')
plt.legend()
plt.show()
plt.grid()
