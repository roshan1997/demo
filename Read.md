```
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

data = pd.DataFrame(data=pd.read_csv("test1.csv"))
#print(data)
data.dropna(inplace=True)
#print(data.values)

x_train = data[["Date","Open","High","Low","Adj Close","Volume"]]
data['Date'] = pd.to_datetime(data.Date)
print(x_train)
y_train = data["Close"]
print(y_train)

#converting into an array
x_train, y_train = np.array(x_train), np.array(y_train)

#LSTM needs 3d array 
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

data1 = pd.DataFrame(data=pd.read_csv("test1.csv"))
#print(data)
data1.dropna(inplace=True)
#print(data.values)
"""
x_test = []
x_test = data1[["Date","Open","High","Low","Adj Close","Volume"]]
data1["Date"] = pd.to_datetime(data1.Date,format='%d-%m-%Y')
print(x_test)
x_test = np.array(x_test)
"""
model = Sequential()
model.add(LSTM(units=100,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=100))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
"""
closing_price = model.predict(x_test)
print(closing_price)
"""


```
