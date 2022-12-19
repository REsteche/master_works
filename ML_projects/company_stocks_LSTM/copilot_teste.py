# create a dataframe of tesla stockmarket values with copilot
import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
# import parameters
yf.pdr_override()
start = dt.datetime(2010,1,1)
end = dt.datetime(2020,1,1)
# load data
df = web.get_data_yahoo(['TSLA'], start=start, end=end)
print(df.head())
# prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
prediction_days = 60
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# build the model
model = Sequential()
# first layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
# second layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# third layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# adding the output layer
model.add(Dense(units=1)) 
model.compile(optimizer='adam', loss='mean_squared_error') # compiling the RNN
model.fit(x_train, y_train, epochs=70, batch_size=32) # fitting the RNN to the training set
# test the model accuracy on existing data
# load Test data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()
test_data = web.get_data_yahoo(['TSLA'], start=test_start, end=test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((df['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)
# make predictions on Test Data
x_test = []
for i in range(60, len(actual_prices) + prediction_days):
    x_test.append(model_inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
# model accuracy
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_stock_price))
print('Root Mean Square Error: %.2f RMSE' % (rmse))
# plot the test predictions
plt.plot(actual_prices, color = 'black', label = 'Tesla Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Tesla Stock Price')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()