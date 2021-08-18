#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
%matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('NSE-TATAGLOBAL.csv')

#print the head
df.head()



# #setting index as date
# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']

# #plot
# plt.figure(figsize=(16,8))
# plt.plot(df['Close'], label='Close Price history')


# #creating dataframe with date and the target variable
# data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

# for i in range(0,len(data)):
#      new_data['Date'][i] = data['Date'][i]
#      new_data['Close'][i] = data['Close'][i]



# #splitting into train and validation
# train = new_data[:987]
# valid = new_data[987:]


# new_data.shape, train.shape, valid.shape



# train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()




# #make predictions
# preds = []
# for i in range(0,248):
#     a = train['Close'][len(train)-248+i:].sum() + sum(preds)
#     b = a/248
#     preds.append(b)


# #calculate rmse
# rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
# rms



# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])


# #setting index as date values
# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']

# #sorting
# data = df.sort_index(ascending=True, axis=0)

# #creating a separate dataset
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

# for i in range(0,len(data)):
#     new_data['Date'][i] = data['Date'][i]
#     new_data['Close'][i] = data['Close'][i]


# #create features
# from fastai.structured import  add_datepart
# add_datepart(new_data, 'Date')
# new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


# new_data['mon_fri'] = 0
# for i in range(0,len(new_data)):
#     if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#         new_data['mon_fri'][i] = 1
#     else:
#         new_data['mon_fri'][i] = 0



# #split into train and validation
# train = new_data[:987]
# valid = new_data[987:]

# x_train = train.drop('Close', axis=1)
# y_train = train['Close']
# x_valid = valid.drop('Close', axis=1)
# y_valid = valid['Close']

# #implement linear regression
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train,y_train)


# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds

# valid.index = new_data[987:].index
# train.index = new_data[:987].index

# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])


# #importing libraries
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))



# #scaling data
# x_train_scaled = scaler.fit_transform(x_train)
# x_train = pd.DataFrame(x_train_scaled)
# x_valid_scaled = scaler.fit_transform(x_valid)
# x_valid = pd.DataFrame(x_valid_scaled)

# #using gridsearch to find the best parameter
# params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
# knn = neighbors.KNeighborsRegressor()
# model = GridSearchCV(knn, params, cv=5)

# #fit the model and make predictions
# model.fit(x_train,y_train)
# preds = model.predict(x_valid)


# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(valid[['Close', 'Predictions']])
# plt.plot(train['Close'])


# from pyramid.arima import auto_arima

# data = df.sort_index(ascending=True, axis=0)

# train = data[:987]
# valid = data[987:]

# training = train['Close']
# validation = valid['Close']

# model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
# model.fit(training)

# forecast = model.predict(n_periods=248)
# forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



# #plot
# plt.plot(train['Close'])
# plt.plot(valid['Close'])
# plt.plot(forecast['Prediction'])



# #importing prophet
# from fbprophet import Prophet

# #creating dataframe
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

# for i in range(0,len(data)):
#     new_data['Date'][i] = data['Date'][i]
#     new_data['Close'][i] = data['Close'][i]

# new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
# new_data.index = new_data['Date']

# #preparing data
# new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

# #train and validation
# train = new_data[:987]
# valid = new_data[987:]

# #fit the model
# model = Prophet()
# model.fit(train)

# #predictions
# close_prices = model.make_future_dataframe(periods=len(valid))
# forecast = model.predict(close_prices)


# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = forecast_valid.values

# plt.plot(train['y'])
# plt.plot(valid[['y', 'Predictions']])



# #importing required libraries
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM

# #creating dataframe
# data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
# for i in range(0,len(data)):
#     new_data['Date'][i] = data['Date'][i]
#     new_data['Close'][i] = data['Close'][i]

# #setting index
# new_data.index = new_data.Date
# new_data.drop('Date', axis=1, inplace=True)

# #creating train and test sets
# dataset = new_data.values

# train = dataset[0:987,:]
# valid = dataset[987:,:]

# #converting dataset into x_train and y_train
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)

# x_train, y_train = [], []
# for i in range(60,len(train)):
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
# x_train, y_train = np.array(x_train), np.array(y_train)

# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))

# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# #predicting 246 values, using past 60 from the train data
# inputs = new_data[len(new_data) - len(valid) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs  = scaler.transform(inputs)

# X_test = []
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
# X_test = np.array(X_test)

# X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# closing_price = model.predict(X_test)
# closing_price = scaler.inverse_transform(closing_price)



# #for plotting
# train = new_data[:987]
# valid = new_data[987:]
# valid['Predictions'] = closing_price
# plt.plot(train['Close'])
# plt.plot(valid[['Close','Predictions']])


