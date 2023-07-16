#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
#Ifeanyichukwu Obi  101 126 269

#COMP4107 A3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import os

cwd = os.getcwd()


# Keras data generator for the hill vallye dataset
class HillValleyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_filepath, batch_size):
        # dataset_filepath is the path to a .data file containing the dataset
        # batch_size is the batch size for the network
        # print(dataset_filepath)

        self.data = pd.read_csv(dataset_filepath, sep=",")

        # print(self.data.shape)

        self.x = self.data.values[:, :-1]
        self.y = self.data.values[:, -1]
        sklearn.preprocessing.minmax_scale(self.x, feature_range=(0, 1), axis=1, copy=False) #normalize data
        # print(self.x.shape)
        # print(self.y.shape)

        # print(self.x)
        # print(self.y)
        self.number_of_images = len(self.x)
        self.batch_size = batch_size
        return

    def __len__(self):
        # batches per epoch is the total number of batches used for one epoch

        batches_per_epoch = self.number_of_images // self.batch_size

        return batches_per_epoch

    def __getitem__(self, index):
        # index is the index of the batch to be retrieved

        batch_of_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_of_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        # x is one batch of data
        # y is the labels associated with the batch
        return batch_of_x, batch_of_y


# A function that creates a keras cnn model to predict whether a sequence has a hill or valley
def hill_valley_cnn_model(dataset_filepath):
    # dataset_filepath is the path to a .data file containing the dataset
    #batch size is 6 bc the next highest common multiple of 606 is 101

    trainingHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Training.data', 6)
    validationHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Validation.data', 6)
    testHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Testing.data', 6)

    model = keras.Sequential()

    model.add(keras.layers.Conv1D(filters=1, kernel_size=5, activation='sigmoid', strides=5,
                                  batch_input_shape=(None, 100, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    training_performance = model.fit(trainingHillValleyGenerator, validation_data=validationHillValleyGenerator, epochs=10, verbose=2)

    validation_performance = model.evaluate(testHillValleyGenerator)
    print(validation_performance)
    return model, training_performance.history['val_loss'][-1], validation_performance[0]


# A function that creates a keras rnn model to predict whether a sequence has a hill or valley
def hill_valley_rnn_model(dataset_filepath):

    trainingHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Training.data', 6)
    validationHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Validation.data', 6)
    testHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Testing.data', 6)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(70, batch_input_shape=(6,100,1)))
    model.add(keras.layers.Dense(70, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    training_performance = model.fit(trainingHillValleyGenerator, validation_data=validationHillValleyGenerator, epochs=10, verbose=2)
    validation_performance = model.evaluate(testHillValleyGenerator)
    print(validation_performance)
    return model, training_performance.history['val_loss'][-1], validation_performance[0]
#print(hill_valley_cnn_model(cwd))
#print(hill_valley_rnn_model(cwd))
