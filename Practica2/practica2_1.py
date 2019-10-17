from scipy.optimize import fmin_tnc as tnc
from ML_UtilsModule import Data_Management
from ML_UtilsModule import Normalization
from matplotlib import pyplot as plt
import numpy as np
import sys

def J(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.dot(np.transpose((np.log(g(np.dot(X, np.transpose(theta)))))), Y)
    var2 = np.dot(np.transpose((np.log(1 - g(np.dot(X, np.transpose(theta)))))), 1 - Y)

    return -((1/m)*(var1 + var2))

def gradient(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.transpose(X)
    var2  = (g(np.dot(X, np.transpose(theta)))-Y)
    return ((1/m) * np.dot(var1, var2))

def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))

def draw_data(X, Y):
    plt.figure()
    pos = np.where(Y == 1) #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='*', c='y')
    pos = np.where(Y == 0) #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='$F$', c='r')

def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = g(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.show()

def training_examples_test_with_theta(training_examples, Y, theta):
    test = g(np.dot(training_examples, np.transpose(theta)))
    test = np.around(test)
    test = np.reshape(test, (np.shape(test)[0], 1))
    mask = (Y == test)
    return (len(Y[mask])/len(Y)) * 100 

data = Data_Management.load_csv(sys.argv[1]) #sys.argv[1])
X, Y, m, n = Data_Management.get_parameters_from_training_examples(data)
draw_data(X, Y)
theta = np.zeros([1, n + 1], dtype=float)
print(theta.shape)
#theta = np.ravel(theta)
X = Data_Management.add_column_left_of_matrix(X) #convention in linear regr

theta = tnc(func=J, x0=theta, fprime=gradient, args=(X,Y))[0]
pinta_frontera_recta(X, Y, theta)
print("Porcentaje de aciertos: " + str(training_examples_test_with_theta(X, Y, theta)) + "%")


