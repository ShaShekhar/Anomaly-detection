import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Input, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    
test_dataframe = pd.read_csv('Data/chart3_modified.csv')
train_dataframe = pd.read_csv('Data/chart2_test.csv')

test_df = test_dataframe[['sequence', 'ALS_Data']]
train_df = train_dataframe[['sequence', 'ALS_Data']]

plt.plot(train_df['sequence'], train_df['ALS_Data'], label='Train data')
plt.plot(test_df['sequence'], test_df['ALS_Data'], label='Test data')
plt.legend()
plt.show()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
# get the distribution of training data
scaler = scaler.fit(train_df[['ALS_Data']])

# normalize the both training and test data w.r.t known distribution
train_df['ALS_Data'] = scaler.transform(train_df[['ALS_Data']])
test_df['ALS_Data'] = scaler.transform(test_df[['ALS_Data']])

seq_size = 5  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train_df[['ALS_Data']], train_df['ALS_Data'], seq_size)
testX, testY = to_sequences(test_df[['ALS_Data']], test_df['ALS_Data'], seq_size)

# print('testX = {}'.format(testX.shape))
# print('testY = {}'.format(testY.shape))

model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, 'saved_model')):
    os.mkdir('saved_model')

model.save('saved_model/my_model')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
###########################
#Anomaly is where reconstruction error is large.
#We can define this value beyond which we call anomaly.
#Let us look at MAE in training prediction

trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
plt.hist(trainMAE, bins=20)
plt.show()

max_trainMAE = 2  #or Define 90% value of max as threshold.

testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=20)
plt.show()
#Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(test_df[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['ALS_Data'] = test_df[seq_size:]['ALS_Data']

plt.plot(anomaly_df['sequence'], anomaly_df['testMAE'])
plt.plot(anomaly_df['sequence'], anomaly_df['max_trainMAE'])
plt.show()
anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

plt.plot(anomaly_df['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomaly_df['ALS_Data']]))))
plt.scatter(anomalies['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomalies['ALS_Data']]))), c='r')
plt.show()
