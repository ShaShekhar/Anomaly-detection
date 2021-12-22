import time
import serial
import pickle
import logging
from datetime import datetime
from threading import Thread
from queue import Queue
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

SAVED_MODEL_PATH = 'saved_model/my_model'

# LOG_FILE_PATH = 'sensor_classifier.log'
# logging.basicConfig(filename=LOG_FILE_PATH, format='%(asctime)s: %(levelname)s: %(message)s',
# 					level=logging.INFO, datefmt='%d/%m/%I %I:%M:%S %p')

class SensorDataStream:
	def __init__(self, serial_path, queue_size=1):
		# initialize the serial data stream along with the boolean
		# used to indicate if the thread should be stopped or not.
		self.stream = serial.Serial(port=serial_path, baudrate=115200,
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
		)
		self.stopped = False
		# initialize the queue used to store sensor data.
		self.Q = Queue(maxsize=queue_size)
		# initialize thread
		self.thread = Thread(target=self.update, args=(serial_path,))
		self.thread.daemon = True

	def start(self):
		# start a thread to read data from sensor
		self.thread.start()
		return self

	def update(self, serial_path):
		try:
            import re
            import numpy as np
            global data
                    
            counter = 0
            five_sample_list = []
            
            # keep looging infinitely
            while True:
                #print('in update while loop')
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    break

                # otherwise, ensure the queue has room in it.
                if not self.Q.full():
                    data = self.stream.readline()

                    decoded_data = data[0:len(data)-2].decode('utf-8')
                    #print(decoded_data)
                    #input('decoded data continue..') #for debug
                    if decoded_data[1] == '1':
                        counter += 1
                        #print('increment counter')
                        pattern = re.compile(r'(\d\d\d+)')
                        matches = pattern.search(decoded_data)
                        a = int(matches.group(1))
                        #print('extracted data {}'.format(a))
                        five_sample_list.append(a)

                        if counter == 5:
                            counter = 0 # reset the counter
                            # print('five_sample_list = {}'.format(five_sample_list))
                            self.Q.put(np.array([five_sample_list]))
                            five_sample_list.clear() # reset the list


		except KeyboardInterrupt:
			self.stream.close()

		except:
			# logging.error(str(e))
			# close and restart the stream.
			self.stream.close()
			self.stream = serial.Serial(port=serial_path, baudrate=115200,
                                        bytesize=serial.EIGHTBITS,
                                        parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE,
                                        timeout=0,
            )

	def read(self):
		# return the data
		# print('in main thread')
		if self.Q.full():
            a = self.Q.get()
            #print('q.get -> {}'.format(a))
		else:
            #print('Queue is empty')
			a = []
		return a

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released.
		self.thread.join()


train_dataframe = pd.read_csv('Data\\chart2_test.csv')
train_df = train_dataframe[['sequence', 'ALS_Data']]

# First get the training data distribution.
scaler = StandardScaler()
scaler = scaler.fit(train_df[['ALS_Data']])

new_model = tf.keras.models.load_model(SAVED_MODEL_PATH)
new_model.summary()

SEQ_SIZE = 5 # Number of time steps to look back
max_trainMAE = 2 # or Define 90% value of max as threshold

def inference(unnorm_data):

    test_norm_data = scaler.transform(unnorm_data)

    testPredict = new_model.predict(test_norm_data)

    testMAE = np.mean(np.abs(testPredict - test_norm_data), axis=1)
    # plt.hist(testMAE, bins=20)
    # plt.show()
    for x in testMAE:
        if x > max_trainMAE:
            print('Dip detected!')

    #Capture all details in a DataFrame for easy plotting
    # anomaly_df = pd.DataFrame(test_df[seq_size:])
    # anomaly_df['testMAE'] = testMAE
    # anomaly_df['max_trainMAE'] = max_trainMAE
    # anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    # anomaly_df['ALS_Data'] = test_df[SEQ_SIZE:]['ALS_Data']

    # #Plot testMAE vs max_trainMAE
    # plt.plot(anomaly_df['sequence'], anomaly_df['testMAE'])
    # plt.plot(anomaly_df['sequence'], anomaly_df['max_trainMAE'])
    # plt.show()
    # anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    # #Plot anomalies
    # plt.plot(anomaly_df['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomaly_df['ALS_Data']]))))
    # plt.scatter(anomalies['sequence'], np.squeeze(scaler.inverse_transform(np.array([anomalies['ALS_Data']]))), c='r')
    # plt.show()

sds = SensorDataStream('COM7')
sds.start()

while True:
    #print('in while loop')
    unnorm_data = sds.read()
    if unnorm_data:
        inference(unnorm_data)
