# Name this file assignment5.py when you submit
import tensorflow as tf
import pandas as pd    
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Functional
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import math

# Keras data generator for the sign language mnist dataset
class ActivityRecognitionDataGenerator(tf.keras.utils.Sequence):

  def __init__(self, filepath_list, batch_size):
    # filepath_list is a list of full file paths to a subset of files containing the data
    # batch_size is the batch size for the network
    # Return nothing    
    
    x=[]
    y=[]
    for file in filepath_list:
      line=np.array(pd.read_csv(file, sep=","))
      pad=4637-line.shape[0]
      a=sklearn.preprocessing.minmax_scale(line, feature_range=(0, 1), axis=1, copy=True)
      a=np.array(a)[:,:-1]
      a=np.pad(a,((0,pad),(0,0)), 'constant', constant_values=(0))
      x.append(a[:1000])
      b=np.pad(line[:,-1],((0,pad)), 'constant', constant_values=(0)).astype(int)
      c=np.zeros((4637,5))
      c[np.arange(b.size), b] = 1
      y.append(c[:1000])
      #y.append(b[:1000])
    x=np.array(x)
    y=np.array(y)

    self.x = x
    self.y = y
    #print(x.shape)
    #print(y.shape)
    self.number_of_datas = len(x)
    self.batch_size = batch_size
    return

  def __len__(self):
    #batches_per_epoch is the total number of batches used for one epoch
    batches_per_epoch = self.number_of_datas // self.batch_size
    return batches_per_epoch

  def __getitem__(self, index):
    # index is the index of the batch to be retrieved

    batch_of_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
    batch_of_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

    # x is one batch of data
    # y is the labels associated with the batch
    return batch_of_x, batch_of_y


# A function that creates a keras rnn model for activity recognition
def activity_rnn_model(training_filepath_list):
  # training_filepath_list is a list of full file paths to files containing the data
  trainingActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)
  validationActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)
  testActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,batch_input_shape=(4,1000,8)))
  model.add(tf.keras.layers.LSTM(70, return_sequences=True))
  model.add(tf.keras.layers.LSTM(70, return_sequences=True))
  model.add(tf.keras.layers.Dense(70, activation="sigmoid"))
  model.add(tf.keras.layers.Dense(5, activation="softmax"))

  adam=tf.keras.optimizers.Adam(learning_rate=0.0001)

  model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['categorical_accuracy'])

  training_performance = model.fit(trainingActivityGenerator, validation_data=validationActivityGenerator, epochs=10, verbose=2)
  validation_performance = model.evaluate(testActivityGenerator)
  print(validation_performance)
  # model is a trained keras rnn model for this task
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance.history['val_loss'][-1], validation_performance[0]


# A function that creates a keras attention-based model for activity recognition
def activity_attention_model(training_filepath_list):
  # training_filepath_list is a list of full file paths to files containing the data
  trainingActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)
  validationActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)
  testActivityGenerator = ActivityRecognitionDataGenerator(training_filepath_list, 4)






  Inputs=tf.keras.Input(shape=(1000,8),batch_size=4)
  x=tf.keras.layers.Masking(mask_value=0., batch_input_shape=(4,1000,8))(Inputs)
  x=tf.keras.layers.Dense(5)(x)
  x, forward_h=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True), merge_mode=None)(x)
  x=tf.keras.layers.Attention(32)([x,forward_h])
  x=tf.keras.layers.Attention(32)([x,forward_h])
  x=tf.keras.layers.Dense(70,activation="sigmoid")(x)
  x=tf.keras.layers.Dense(5, activation="softmax")(x)
  model=keras.Model(inputs=Inputs,outputs=x)
  adam=tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
  print(model.summary())


  training_performance = model.fit(trainingActivityGenerator, validation_data=validationActivityGenerator, epochs=4, verbose=2)
  validation_performance = model.evaluate(testActivityGenerator)
  print(validation_performance)
  # model is a trained keras rnn model for this task
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance.history['val_loss'][-1], validation_performance[0]

def main():
  path1='./Datasets_Healthy_Older_People/S1_Dataset/'
  filepaths1=os.listdir(path1)
  path2='./Datasets_Healthy_Older_People/S2_Dataset/'
  filepaths2=os.listdir(path2)
  filepaths=[path1 + e for e in filepaths1] + [path2 + e for e in filepaths2]

  print(activity_rnn_model(filepaths))
  print(activity_attention_model(filepaths))

main()