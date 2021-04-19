#!/usr/bin/env python
# coding: utf-8

# In[348]:


import pandas_datareader.data as reader
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# In[349]:


end = dt.datetime.now()
start = dt.date(end.year - 5,end.month,end.day)
kryptolist = ['BTC-USD', 'ETH-USD', 'XRP-USD','LTC-USD']


# In[350]:


df = reader.get_data_yahoo(kryptolist,start,end)['Adj Close']


# In[351]:


df


# In[352]:


df.plot()
plt.show()


# In[353]:


daily_returns = df.pct_change().dropna(axis=0)


# In[354]:


daily_returns


# In[355]:


daily_cum_returns = (daily_returns + 1).cumprod() -1


# In[356]:


daily_cum_returns


# In[357]:


colors = ['r','k','b','g','y']
daily_cum_returns.plot(color=colors, figsize=(11,6))
plt.title('Cumulative returns of the largest cryptocurrencies')
plt.show()


# In[358]:


fig, axs = plt.subplots(2, 2, figsize=(14,8),gridspec_kw={'hspace': 0.2,'wspace': 0.1})

axs[0,0].plot(df['BTC-USD'], c='r')
axs[0,0].set_title('BTC')
axs[0,1].plot(df['ETH-USD'], c='k')
axs[0,1].set_title('ETH')
axs[1,0].plot(df['XRP-USD'], c='b')
axs[1,0].set_title('XRP')
axs[1,1].plot(df['LTC-USD'], c='g')
axs[1,1].set_title('LTC')
plt.show()


# In[359]:


fig, axs = plt.subplots(2, 2, figsize=(14,8),gridspec_kw={'hspace': 0.2,'wspace': 0.1})

axs[0,0].plot(daily_returns['BTC-USD'], c='r')
axs[0,0].set_title('BTC')
axs[0,0].set_ylim([-0.6,0.6])
axs[0,1].plot(daily_returns['ETH-USD'], c='k')
axs[0,1].set_title('ETH')
axs[0,0].set_ylim([-0.6,0.6])
axs[1,0].plot(daily_returns['XRP-USD'], c='b')
axs[1,0].set_title('XRP')
axs[0,0].set_ylim([-0.6,0.6])
axs[1,1].plot(daily_returns['LTC-USD'], c='g')
axs[1,1].set_title('LTC')
axs[0,0].set_ylim([-0.6,0.6])
plt.show()


# In[360]:


fig, axs = plt.subplots(2, 2, figsize=(14,8),gridspec_kw={'hspace': 0.2,'wspace': 0.1})

axs[0,0].hist(daily_returns['BTC-USD'],bins=100, color = 'r',range=(-0.2, 0.2))
axs[0,0].set_title('BTC')
axs[0,1].hist(daily_returns['ETH-USD'],bins=100, color = 'k',range=(-0.2, 0.2))
axs[0,1].set_title('ETH')
axs[1,0].hist(daily_returns['XRP-USD'],bins=100, color = 'b',range=(-0.2, 0.2))
axs[1,0].set_title('XRP')
axs[1,1].hist(daily_returns['LTC-USD'],bins=100, color = 'g',range=(-0.2, 0.2))
axs[1,1].set_title('LTC')
plt.show()


# In[361]:


daily_returns.boxplot()
plt.title('Boxplot of daily returns with outliers')
plt.show()


# In[362]:


daily_returns.boxplot(showfliers=False)
plt.title('Boxplot of daily returns without outliers')


# In[363]:


daily_returns.corr()


# In[271]:


sns.heatmap(daily_returns.corr(), vmin=0, vmax=1, annot=True)
plt.show()


# In[364]:


from fbprophet import Prophet
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


# In[381]:


end1 = dt.date(end.year - 1, end.month,end.day) #1 year back
end2 = dt.date(end.year,end.month - 3, end.day) #6 months back <-- Works only with numbers 1-3, 4-12 I've got error 
start1 = dt.date(end1.year - 3, end.month, end.day)

///////////////////////////////////////////////////////////////////////////////
#Code With Error:
end1 = dt.date(end.year - 1, end.month,end.day) #1 year back
end2 = dt.date(end.year,end.month - 6, end.day) #6 months back
start1 = dt.date(end1.year - 3, end.month, end.day)

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-382-04b20b04d755> in <module>
      1 end1 = dt.date(end.year - 1, end.month,end.day) #1 year back
----> 2 end2 = dt.date(end.year,end.month - 6, end.day) #6 months back
      3 start1 = dt.date(end1.year - 3, end.month, end.day)

ValueError: month must be in 1..12

//////////////////////////////////////////////////////////////////////////////

# In[366]:


df1 = reader.get_data_yahoo(kryptolist,start1,end1)['Adj Close']
df2 = reader.get_data_yahoo(kryptolist,start1,end2)['Adj Close']


# In[367]:


modelfb1 = Prophet()
df1 = df1.reset_index()
df1[['ds','y']] = df1[['Date','BTC-USD']]
modelfb1.fit(df1)


# In[368]:


modelfb2 = Prophet()
df2 = df2.reset_index()
df2[['ds','y']] = df2[['Date','BTC-USD']]
modelfb2.fit(df2)


# In[369]:


future = modelfb1.make_future_dataframe(periods=365)


# In[370]:


future1 = modelfb2.make_future_dataframe(periods=90)


# In[371]:


forecast1 = modelfb1.predict(future)
forecast2 = modelfb2.predict(future1)


# In[372]:


modelfb1.plot(forecast1)
plt.title('Backtest of BTC 1 Year Prediction')
plt.show()


# In[373]:


modelfb1.plot(forecast2)
plt.title('Backtest of BTC 3 Months Prediction')
plt.show()


# In[374]:


modelfb3 = Prophet()
df = df.reset_index()
df[['ds','y']] = df[['Date','BTC-USD']]
modelfb3.fit(df)


# In[375]:


future3 = modelfb3.make_future_dataframe(periods=241)


# In[376]:


forecast3 = modelfb3.predict(future3)


# In[377]:


modelfb3.plot(forecast3)
plt.title('BTC in the future 3 months')
plt.show()


# In[378]:


import statsmodels.api as sm


# In[ ]:





# In[ ]:





# In[379]:


y = daily_returns['BTC-USD']
X1 = daily_returns[['ETH-USD']]
X2 = daily_returns[['ETH-USD','XRP-USD']]
X3 = daily_returns[['ETH-USD','XRP-USD','LTC-USD']]

X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)
X3 = sm.add_constant(X3)


# In[ ]:

---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-396-0c2468c93995> in <module>
      4 X3 = daily_returns[['ETH-USD','XRP-USD','LTC-USD']]
      5 
----> 6 X1 = sm.add_constant(X1)
      7 X2 = sm.add_constant(X2)
      8 X3 = sm.add_constant(X3)

~/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tools/tools.py in add_constant(data, prepend, has_constant)
    294     if _is_using_pandas(data, None) or _is_recarray(data):
    295         from statsmodels.tsa.tsatools import add_trend
--> 296         return add_trend(data, trend='c', prepend=prepend, has_constant=has_constant)
    297 
    298     # Special case for NumPy

~/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tsa/tsatools.py in add_trend(x, trend, prepend, has_constant)
     95                 except:
     96                     return False
---> 97             col_const = x.apply(safe_is_const, 0)
     98         else:
     99             ptp0 = np.ptp(np.asanyarray(x), axis=0)

~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in apply(self, func, axis, broadcast, raw, reduce, result_type, args, **kwds)
   6898             If an array is passed, it must be the same length as the data. The
   6899             list can contain any of the other types (except list).
-> 6900             Keys to group by on the pivot table column.  If an array is passed,
   6901             it is being used as the same manner as column values.
   6902         aggfunc : function, list of functions, dict, default numpy.mean

~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/apply.py in <module>
      7 from pandas._config import option_context
      8 
----> 9 from pandas._typing import Axis, FrameOrSeriesUnion
     10 from pandas.util._decorators import cache_readonly
     11 

ImportError: cannot import name 'FrameOrSeriesUnion' from 'pandas._typing' (/Users/lukasz/opt/anaconda3/lib/python3.7/site-packages/pandas/_typing.py)


