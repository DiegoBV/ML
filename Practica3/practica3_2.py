from ML_UtilsModule import Data_Management
from scipy.io import loadmat
from scipy.optimize import fmin_tnc as tnc
from matplotlib import pyplot as plt
import numpy as np


def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))


def propagation(X, theta1, theta2):
    hiddenLayer = g(np.dot(X, np.transpose(theta1)))      
    hiddenLayer = Data_Management.add_column_left_of_matrix(hiddenLayer)
    
    outputLayer = g(np.dot(hiddenLayer, np.transpose(theta2)))
    
    return outputLayer
    
    
def checkLearned(y, outputLayer):
    
    maxIndexV = np.argmax(outputLayer, axis = 1) + 1
    
    checker = (y[:,0] == maxIndexV) 
    count = np.size(np.where(checker == True)) 
    fin = count/np.shape(y)[0] * 100
    return fin



weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

X, y = Data_Management.load_mat("ex3data1.mat")

X = Data_Management.add_column_left_of_matrix(X) #añadida culumna de 1s

outputLayer = propagation(X, theta1, theta2)

print("Precision de la red neuronal: " + str(checkLearned(y, outputLayer)) + " %")