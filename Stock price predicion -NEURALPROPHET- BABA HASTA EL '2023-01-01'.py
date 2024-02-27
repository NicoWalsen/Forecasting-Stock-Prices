#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[ ]:


import os    


# In[29]:


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[4]:


pip install neuralprophet


# In[31]:


from neuralprophet import NeuralProphet


# In[81]:


import yfinance as yf


# In[82]:


import pandas as pd


# In[83]:


import matplotlib.pyplot as plt


# In[84]:


stock_symbol = 'BABA'#ALIBABA


# In[108]:


start_date = '2015-01-01'


# In[109]:


end_date = '2023-12-31'


# In[110]:


stock_data = yf.download(stock_symbol,start=start_date, end=end_date)


# In[111]:


print(stock_data.head())


# In[112]:


stock_data.to_csv('stock_data.csv')


# In[113]:


stocks = pd.read_csv('stock_data.csv')


# In[114]:


stocks.dtypes


# In[115]:


stocks['Date'] = pd.to_datetime(stocks['Date'])


# In[116]:


stocks = stocks[['Date','Close']]


# In[117]:


stocks


# In[118]:


stocks.dtypes


# In[119]:


stocks.columns = ['ds','y']


# In[120]:


stocks


# In[2]:


plt.plot(stocks['ds'],stocks['y'],label='actual', c='g')
plt.grid()


# In[122]:


plt.show()


# In[123]:


#TRAIN THE MODEL


# In[124]:


model = NeuralProphet()


# In[125]:


model.fit(stocks)


# In[126]:


future = model.make_future_dataframe(stocks, periods = 365)


# In[127]:


forecast = model.predict(future)


# In[128]:


forecast


# In[129]:


actual_prediction = model.predict(stocks)


# In[130]:


plt.plot(actual_prediction['ds'],actual_prediction['yhat1'],label="prediction_actual",c='r')
plt.plot(forecast['ds'],forecast['yhat1'],label="forecast",c='b')
plt.plot(stocks['ds'],stocks['y'],label='actual', c='g')
plt.legend()
plt.show()
plot.grid()


# In[ ]:





# In[ ]:




