from scipy.optimize import fmin_tnc as tnc
from ML_UtilsModule import Data_Management
from ML_UtilsModule import Normalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
import sys

learning_rate = 1.0

def polinomial_features(X, grado):
    poly = pf(grado)
    return (poly, poly.fit_transform(X))

def J(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.dot(np.transpose((np.log(g(np.dot(X, np.transpose(theta)))))), Y)
    var2 = np.dot(np.transpose((np.log(1 - g(np.dot(X, np.transpose(theta)))))), (1 - Y))
    var3 = (learning_rate/(2*m)) * np.sum(theta[1:]**2)
    return (((-1/m)*(var1 + var2)) + var3)

def gradient(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.transpose(X)
    var2  = g(np.dot(X, np.transpose(theta)))-Y
    
    theta = np.c_[[0], theta[:, 1:]]
    var3 = (learning_rate/m) * theta
    return ((1/m) * np.dot(var1, var2)) + np.transpose(var3)

def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))

def draw_data(X, Y):
    pos = np.where(Y == 1) #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='*', c='y')
    pos = np.where(Y == 0) #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='$F$', c='r')

def draw_decision_boundary(theta, X, Y, poly):
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))
    
    sigm = g(poly.fit_transform(np.c_[ xx1.ravel(), xx2.ravel()]).dot(theta))
    sigm = sigm.reshape(xx1.shape)

    plt.contour(xx1, xx2, sigm, [0.5], linewidths = 1, colors = 'g')

def draw(theta, X, Y, poly):
    plt.figure()
    draw_data(X, Y)
    draw_decision_boundary(theta, X, Y, poly)
    plt.show()

data = Data_Management.load_csv(sys.argv[1]) #sys.argv[1])
X, Y, m, n = Data_Management.get_parameters_from_training_examples(data)
poly, X_poly = polinomial_features(X, int(sys.argv[2]))
theta = np.zeros([1, np.shape(X_poly)[1]], dtype=float)

theta = tnc(func=J, x0=theta, fprime=gradient, args=(X_poly,Y))[0]


draw(theta, X, Y, poly)
