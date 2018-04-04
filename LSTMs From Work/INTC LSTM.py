import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
#for checking for file existance
import os
#from pathlib import Path
drop_value = 0.0001
epochs_to_run = 4
#function to get stock data from the internet
def get_stock_data(stock_name, normalized = 0):
    #url = 'http://chart.finance.yahoo.com/table.csv?s=%s&a=11&b=15&c=1981&d=29&e=10&f=2016&g=d&ignore=.csv' % stock_name
    url = '%s.csv' % stock_name
    col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stocks = pd.read_csv(url, header=0, names=col_names)
    df = pd.DataFrame(stocks)
    date_split = df['Date'].str.split('-').str
    #Trying to make the closing price as the last feature
    closing_price_column = df['Close']
    df['Year'], df['Month'], df['Day'] = date_split
    df['Volume'] = df["Volume"] / 10000
    df.drop(df.columns[[0,1,2,3,4,5,6]], axis=1, inplace=True)
    #Trying to append the closing price
    df['Close'] = closing_price_column
    return df

#Loading stock data
stock_name = "INTC"
df = get_stock_data(stock_name, 0)
#print(df.head())
#saving the data to the file for future use
today = datetime.date.today()
file_name = stock_name + '_stock_%s.csv' % today
#file_path = Path(file_name)
#if file_path.exists() :
if os.path.isfile(file_name):
    os.remove(file_name)
else:
    print("Error, no file found!")
df.to_csv(file_name)
#modifying data to ones and zeros
def convert_dataframe_to_numbers(df):
    #modifying new data
    df['Year'] = pd.to_numeric(df['Year'])#df['Year'].convert_objects(convert_numeric=True)
    df['Month'] = pd.to_numeric(df['Month'])#df['Month'].convert_objects(convert_numeric=True)
    df['Day'] = pd.to_numeric(df['Day'])#df['Day'].convert_objects(convert_numeric=True)
    
    df['Year'] = df['Year'] / 1000
    df['Day'] = df['Day'] / 10
    df['Close'] = df['Close'] / 10
    df['Month'] = df['Month'] / 10
    return df

df = convert_dataframe_to_numbers(df)

#print(df.head(5))#temporaty check
#this part would be good for forming prediction imput!!!
#Updated Load_Data function from lstm.py, configured to accept any ammont of features.  It is set to calculate the last feature as a result.
def load_data(stock, seq_len):
    #getting the total ammount of features
    ammont_of_features = len(stock.columns)
    #converting dataframe to matrix
    data = stock.as_matrix() #pd.DataFrame(stock)
    #total number of rows
    sequence_length = seq_len + 1
    #resulting output
    result = []
    #iterating through the matrix to form the result array
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    #converting results array to numpy array
    result = np.array(result)
    #cutting off the results and reshaping them at a given row (1.11% as testing data 99.99% as training data)  change the 0.9 to change the ammount for training
    row = round(0.01 * result.shape[0])
    #training array cutting matrix to have just the rows
    #train = result[:int(row), :]
    test = result[:int(row), :]
    x_test = test[:, :-1]
    y_test = test[:, -1][:,-1]
    x_train = result[int(row):, :-1]
    y_train = result[int(row):, -1][:,-1]
    #messing with x's to have just the date for input
    x_test = np.delete(x_test, np.s_[-1:], axis=2)
    x_train = np.delete(x_train, np.s_[-1:], axis=2)

    """
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    """
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], ammont_of_features - 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], ammont_of_features - 1))

    return [x_train, y_train, x_test, y_test]
#building model functions
'''
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time: ", time.time() - start)
    return model
'''
#Setting x and y for training and testing
#window is the sequence length of samples
window = 22
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
#print('X_train', X_train.shape)
#print('y_train', y_train.shape)
#print('X_test', X_test.shape)
#print('y_test', y_test.shape)

#working with the next model function
def build_model2(layers):
    #the dropout value
    dropout_value = drop_value
    #setting the model type
    model = Sequential()
    #adding model layers
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(dropout_value))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(dropout_value))
    #old keras api for dense
    #model.add(Dense(16, init='uniform',activation='relu'))
    #model.add(Dense(1,init='uniform',activation='linear'))
    #new API
    model.add(Dense(16, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='uniform'))
    #compiling model
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model
#Loading the model sequence structure
# model = build_model([2,lag,1])
#model = build_model2([3,window,1])#[y of neural net, x of neural net,  idk what one is for yet]
#trying out something new
model = build_model2([X_test.shape[2],window,1])
#once we have a saved model lets load it instead of all the other crap!
#executing the model & RMS/RMSE results
model.fit(
    X_train,#inputs
    y_train,#outputs
    batch_size=512,#ammount of samples evaluated at a time
    epochs=epochs_to_run,#make sure to increase number of epochs to 5000
    validation_split=0.01,
    verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

#print(X_test[-1])
diff=[]
ratio=[]
#predition and evaluation portion
p = model.predict(X_test)
#print(p)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
#prediction vs real result
#Why can't  I use plt?
#****import matplotlib.pyplot as plt2
model.save(('%s_drop0.0001_model.h5' % stock_name))

'''

plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label="y_test")
plt.legend(loc='upper left')
plt.show()
'''