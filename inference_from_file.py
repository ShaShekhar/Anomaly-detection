import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

test_dataframe = pd.read_csv('Data/chart3_modified.csv')
train_dataframe = pd.read_csv('Data/chart2_test.csv')

test_df = test_dataframe[['sequence', 'ALS_Data']]
train_df = train_dataframe[['sequence', 'ALS_Data']]

# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(train_df[['ALS_Data']])

train_df['ALS_Data'] = scaler.transform(train_df[['ALS_Data']])
test_df['ALS_Data'] = scaler.transform(test_df[['ALS_Data']])

new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

seq_size = 5 # Number of time steps to look back
max_trainMAE = 2 # or Define 90% value of max as threshold

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

testX, testY = to_sequences(test_df[['ALS_Data']], test_df['ALS_Data'], seq_size)

testPredict = new_model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=20)
plt.show()
#Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(test_df[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['ALS_Data'] = test_df[seq_size:]['ALS_Data']

#Plot testMAE vs max_trainMAE
plt.plot(anomaly_df['sequence'], anomaly_df['testMAE'])
plt.plot(anomaly_df['sequence'], anomaly_df['max_trainMAE'])
plt.show()
anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

#Plot anomalies
plt.plot(anomaly_df['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomaly_df['ALS_Data']]))))
plt.scatter(anomalies['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomalies['ALS_Data']]))), c='r')
plt.show()
