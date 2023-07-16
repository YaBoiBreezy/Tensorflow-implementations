import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

#COMP 4107 assignment 2
#Alexander Breeze  101 143 291
#Michael Balcerzak 101 071 699
#Ifeanyichukwu Obi 101 126 269

# A function that implements a keras model with the sequential API following the provided description
def sequential_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(12, input_shape=(4,), activation="relu"))
    model.add(keras.layers.Dense(6, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="cross_entropy")

    # A keras model
    return model


# A function that implements a keras model with the functional API following the provided description
def functional_model():
    input = keras.Input(shape=(5,))
    layer_2 = keras.layers.Dense(8, activation="relu")(input)
    layer_3 = keras.layers.Dense(8, activation="relu")(layer_2)
    layer_4 = keras.layers.Dense(4, activation="relu")(layer_3)
    output = keras.layers.Dense(1, activation="relu")(layer_4)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer='sgd', loss='mse')

    # A keras model
    return model


# A function that creates a keras model to predict concrete compressive strength
def compressive_strength_model(filepath):
    # filepath is the path to an xls file containing the dataset
    WS = pd.read_excel(filepath)
    data = np.array(WS)
    x = data[:, :-1]
    x = x / x.max(axis=0)
    y = data[:, -1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="relu"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")

    res=model.fit(x=x, y=y, validation_split=0.2, epochs=38, verbose=0)
    #getting performance on validation set, NOT test set like in 4d, as per the instructions.
    validation_performance = res.history['val_loss'][-1]

    # model is a trained keras model for predicting compressive strength
    # validation_performance is the performance of the model on a validation set
    return model, validation_performance

# 4a
def compressive_strength_model4a(filepath):
    WS = pd.read_excel(filepath)
    data = np.array(WS)
    print(data.shape)
    x = data[:, :-1]
    x = x / x.max(axis=0)
    y = data[:, -1]

    hidden_layer = [4, 8, 12, 20, 40]
    for H1 in hidden_layer:
        for H2 in hidden_layer:
            model = keras.Sequential()
            model.add(keras.layers.Dense(H1, input_shape=(8,), activation="sigmoid"))
            model.add(keras.layers.Dense(H2, activation="sigmoid"))
            model.add(keras.layers.Dense(1, activation="linear"))
            optimizer = tf.keras.optimizers.SGD()
            model.compile(optimizer=optimizer, loss="mse")
            print(f"hidden={H1}, hidden2={H2} ")
            model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)


def compressive_strength_model4b(filepath):
    WS = pd.read_excel(filepath)
    data = np.array(WS)
    print(data.shape)
    x = data[:, :-1]
    x = x / x.max(axis=0)
    y = data[:, -1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=10, verbose=2)


def compressive_strength_model4c(filepath):
    WS = pd.read_excel(filepath)
    data = np.array(WS)
    print(data.shape)
    x = data[:, :-1]
    x = x / x.max(axis=0)
    y = data[:, -1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(x=x, y=y, validation_split=0.2, epochs=50, verbose=2)



def compressive_strength_model4d(filepath):
    # filepath is the path to an xls file containing the dataset
    WS = pd.read_excel(filepath)
    data = np.array(WS)
    print(data.shape)
    x = data[:, :-1]
    x = x / x.max(axis=0)
    y = data[:, -1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(8,), activation="relu"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(20, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="linear"))
    print(model.summary())
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="mse")

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    # x_train=np.array(x_train[:5])
    # y_train=np.array(y_train[:5])

    model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=38, verbose=2)
    validation_performance = model.evaluate(x_test, y_test)
    print(validation_performance)

    return model, validation_performance

print(compressive_strength_model('./Concrete_Data.xls'))
compressive_strength_model4d('./Concrete_Data.xls')
