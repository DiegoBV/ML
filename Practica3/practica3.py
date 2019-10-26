from ML_UtilsModule import Data_Management
from scipy.optimize import fmin_tnc as tnc
from matplotlib import pyplot as plt
import numpy as np

learning_rate = 0.1


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

def oneVSAll(X, y, num_etiquetas, reg):
    """
    oneVsAll entrena varios clasificadores por regresión logística con término
    de regularización ’reg’ y devuelve el resultado en una matriz, donde
    la fila i−ésima corresponde al clasificador de la etiqueta i−ésima
    """
    clasificadores = np.empty((num_etiquetas, np.shape(X)[1])) 
    theta = np.random.standard_normal((1, np.shape(X)[1])) #TODO: init with random values

    mask = np.empty((num_etiquetas, np.shape(y)[0]), dtype=bool)

    #TODO: VECTORIZAR SI SE PUEDE UWU       
    for i in range(num_etiquetas):
        mask[i, :] = (y[:, 0]% num_etiquetas == i)
        clasificadores[i] = tnc(func=J, x0=theta, fprime=gradient, args=(X, np.reshape(mask[i], (np.shape(X)[0], 1))))[0]
    return clasificadores

def checkLearned(X, y, clasificadores):
    result = checkNumber(X, clasificadores)
    
    maxIndexV = np.argmax(result, axis = 1);

    checker = ((y[:,0]%np.shape(clasificadores)[0]) == maxIndexV) 
    count = np.size(np.where(checker == True)) 
    fin = count/np.shape(y)[0] * 100
    
    return fin

def checkNumber(X, clasificadores):
    result = np.zeros((np.shape(X)[0],np.shape(clasificadores)[0]))

    result[:] = g(np.dot(X, np.transpose(clasificadores[:]))) #result[0] = todo lo que piensa de cada numero respecto a si es 0
    return result

X, y = Data_Management.load_mat("ex3data1.mat")
clasificadores = oneVSAll(X, y, 10, 2)

print (str (checkLearned(X, y, clasificadores)) + " %")

Data_Management.draw_random_examples(X)
plt.show()