# import required packages


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow.
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pickle import dump,load
from keras.layers import LSTM
from keras.layers import Dense,BatchNormalization, Conv2D, MaxPool2D
from keras.layers import Dropout
from keras.models import Sequential, Model
from keras.utils import to_categorical

def dataframe():
    data = pd.read_csv('data/q2_dataset.csv')
    #Creating column names
    column_list = ['volume_day_3','open_day_3','high_day_3','low_day_3','volume_day_2','open_day_2','high_day_2','low_day_2','volume_day_1','open_day_1','high_day_1','low_day_1','present']
    timeseries = []
    idx_list = []
    for i in range(3,1259):
      sample = data.iloc[i-3,2:].to_list() + data.iloc[i-2,2:].to_list() + data.iloc[i-1,2:].to_list()
      sample.append(data.iloc[i,3])
      timeseries.append(sample)
      idx_list.append(data.index[i])
    
    return(pd.DataFrame(data=timeseries, columns=column_list, index=idx_list))

if __name__ == "__main__": 
        
#    timeseries_df = dataframe()
#    #scaling data using standard scaler
#    scaled = preprocessing.StandardScaler()
#    scaled.fit(timeseries_df.iloc[:,:-1])
#    dump(scaled, open('models/scaled.pkl', 'wb'))
#    #Scaling Data
#    scaled_label = preprocessing.StandardScaler()
#    scaled_label.fit(timeseries_df.iloc[:,-1].to_numpy().reshape((timeseries_df.shape[0],1)))
#    dump(scaled_label, open('models/scaled_label.pkl','wb'))
#    training, testing = train_test_split(timeseries_df, test_size=0.3)
#    
#    training.to_csv('data/train_data_RNN.csv', index=True)
#    testing.to_csv('data/test_data_RNN.csv', index=True)
    
    #Importing created pickle files and dataset
    scaled = load(open('models/scaled.pkl', 'rb'))
    scaled_label = load(open('models/scaled_label.pkl', 'rb'))
    dataset = pd.read_csv('data/train_data_RNN.csv', index_col=0)
    
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    
    X = scaled.fit_transform(X)
    y = scaled_label.fit_transform(y.to_numpy().reshape(y.shape[0],1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    
    X_train = X_train.reshape(X_train.shape[0],3,4)
    X_test = X_test.reshape(X_test.shape[0],3,4)
    
    model = Sequential()
    model.add(LSTM(units = 100,return_sequences=True, input_shape=(3,4)))
    model.add(Dropout(rate = 0.4))
    model.add(LSTM(units = 100, return_sequences=True))
    model.add(Dropout(rate = 0.4))
    model.add(LSTM(units = 100))
    model.add(Dropout(rate = 0.4))
    model.add(Dense(units = 1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.summary()
    final_model = model.fit(X_train,y_train,batch_size=128,epochs=1000,validation_data=(X_test,y_test))
    
    loss = final_model.history['loss']
    val_loss = final_model.history['val_loss']
    epochs = range(1000)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss',color='red')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    model.save('models/20812666_RNN_model.h5')
