#from keras.models import Sequential, Model
#from keras.layers import Dense
import numpy as np


# A function simulating an artificial neuron
def artificial_neuron(x, w):
    # x is a list of inputs of length n
    # w is a list of inputs of length n
    if (len(x) != len(w)):
        return "ERROR: INVALID SIZED INPUTS"
    f = 0
    for i in range(len(x)):
        f += x[i] * w[i]
    e = 2.718281828459
    output = ((e ** f) - (e ** (-1 * f))) / ((e ** f) + (e ** (-1 * f)))
    # output is the output from the neuron
    return output


# A function performing gradient descent
def gradient_descent(f, df, alpha, x0):
    # f is a function that takes as input a list of length n
    # df is the gradient of f; it is a function that takes as input a list of length n
    # alpha is the learning rate
    # x0 is an initial guess for the input minimizing f

    if not callable(f) or not callable(df):
        return "WRONG format for f and df"

    y = 100000000
    max_iter = 1000
    for _ in range(max_iter):
        x0 = x0 - alpha * df(x0)
        if f(x0) < y:
            y = f(x0)
        else:
            break

    argmin_f = x0
    min_f = f(x0)

    # argmin_f is the input minimizing f
    # min_f is the value of f at its minimum
    return argmin_f, min_f


# A function computing the singular value decomposition of a matrix
def svd(A):
    # A is a matrix

    U, S, V = np.linalg.svd(A)

    left_vectors = U
    values = S
    right_vectors = V.T
    # values is a list (not a diagonal matrix) of singular values sorted in descreasing order
    # left_vectors is a matrix whose columns are the left singular values of A
    # right_vectors is a matrix whose columns are the right singular values of A
    return values, left_vectors, right_vectors


# A function that returns a keras model
def keras_model():
    model = Sequential()
    model.add(Dense(2, input_dim=1, activation='relu'))
    print(model.summary())

    # A keras model
    return model


print(artificial_neuron([1, 2], [0.12, 0.16]))

print("D: ", svd(np.array([[6, 4, 6], [1, 1, 1], [2, 3, 9]]))[0])
print("U: ", svd(np.array([[6, 4, 6], [1, 1, 1], [2, 3, 9]]))[1])
print("V: ", svd(np.array([[6, 4, 6], [1, 1, 1], [2, 3, 9]]))[2])

print("D: ", svd([[5, 1, 2], [7, 4, 9], [1, 6, 3]])[0])
print("U: ", svd([[5, 1, 2], [7, 4, 9], [1, 6, 3]])[1])
print("V: ", svd([[5, 1, 2], [7, 4, 9], [1, 6, 3]])[2])


def quad(x):
    return x[0] ** 2 + 5 * x[0] + 5


def grad(x):
    return 2 * x[0] + 5


print(gradient_descent(quad, grad, 0.1, np.array([3])))

#keras_model()
