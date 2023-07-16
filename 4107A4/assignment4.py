# Name this file assignment4.py when you submit
#Alexander Breeze   101 143 291
#Michael Balcerzak  101 071 699
#Ifeanyichukwu Obi  101 126 269

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import os
from PIL import Image

cwd = os.getcwd()



# Keras data generator for the sign language mnist dataset
class SignLanguageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_filepath, use_rows, batch_size):
        # dataset_filepath is the path to a .csv file containing the dataset
        # use_rows is a subset of the rows in the file to be used
        # batch_size is the batch size for the network

        # Return nothing
        data = np.array(pd.read_csv(dataset_filepath, sep=","))
        data = np.delete(data, 0, 1)
        data = data[use_rows, :]
        data = np.reshape(data, (len(use_rows), 28**2))
        data=sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)
        sc=sklearn.preprocessing.StandardScaler()
        sc.fit(data)
        data = np.reshape(data, (len(use_rows), 28, 28, 1))

        self.data = data
        self.number_of_images = len(use_rows)
        self.batch_size = batch_size
        return

    def __len__(self):
        # batches_per_epoch is the total number of batches used for one epoch
        batches_per_epoch = self.number_of_images // self.batch_size
        return batches_per_epoch

    def __getitem__(self, index):
        # index is the index of the batch to be retrieved
        x = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        # x is one batch of data
        # y is the same batch of data
        return x, y


# A function that creates a keras cnn autoencoder model for images of American sign language
def sign_language_autoencoder(training_data_filepath):
    trainingSignGenerator = SignLanguageDataGenerator(training_data_filepath + "/sign_mnist_train.csv", range(0, 20), 6)
    validationSignGenerator = SignLanguageDataGenerator(training_data_filepath + "/sign_mnist_train.csv", range(20, 30),
                                                        6)
    testSignGenerator = SignLanguageDataGenerator(training_data_filepath + "/sign_mnist_test.csv", range(9), 6)
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))
    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(16, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', strides=2, padding='same', kernel_regularizer=keras.regularizers.l2(0.001))])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(8, kernel_size=3, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2DTranspose(16, kernel_size=3, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2DTranspose(32, kernel_size=3, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=3, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.l2(0.001))])
    encoded = encoder(input_img)
    decoded = decoder(encoded)

    model = tf.keras.Model(input_img, decoded)
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer, loss='mse')
    #print(model.summary())

    training_performance = model.fit(trainingSignGenerator, validation_data=validationSignGenerator, epochs=25,
                                     verbose=2)
    test_performance = model.evaluate(testSignGenerator)
    #print(test_performance)

    return model, training_performance.history['loss'][-1], training_performance.history['val_loss'][-1]

def main():
    obj = SignLanguageDataGenerator("./sign_mnist_train.csv", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    # print(obj.__len__())
    # print(obj.__getitem__(0))
    # print(obj.__getitem__(1))
    # print(obj.__getitem__(2))
    print(sign_language_autoencoder(cwd))
    # compressive_strength_mode3a(cwd)
    # compressive_strength_mode3b(cwd)
    # compressive_strength_mode3c(cwd)
    #expert_encoding_Testing4a(cwd)


# main()

