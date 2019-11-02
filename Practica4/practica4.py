from ML_UtilsModule import Data_Management
from scipy.optimize import fmin_tnc as tnc
from scipy.io import loadmat
from matplotlib import pyplot as plt
import displayData
import numpy as np

learning_rate = 1

def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))

def derivada_de_G(z):
    result = g(z) * (1 - g(z))
    return result

def pesos_aleat(L_in, L_out):
    pesos = np.random.uniform(-0.12, 0.12, (L_out, 1+L_in))
    
    return pesos

def transform_y(y, num_etiquetas):
    mask = np.empty((num_etiquetas, np.shape(y)[0]), dtype=bool)
    for i in range( num_etiquetas):
        mask[i, :] = ((y[:, 0] + num_etiquetas - 1) % num_etiquetas == i) 
        #codificado con el numero 1 en la posicion 0 y el numero 0 en la posicion 9
    
    mask = mask * 1

    return np.transpose(mask)

def J(X, y, a3, num_etiquetas, theta1, theta2):
    m = np.shape(X)[0]
    aux1 = -y * (np.log(a3))
    aux2 = (1 - y) * (np.log(1 - a3))
    aux3 = aux1 - aux2
    aux4 = np.sum(theta1**2) + np.sum(theta2**2)
    print (aux4)
    return (1/m) * np.sum(aux3) + (learning_rate/(2*m)) * aux4

def propagation(a1, theta1, theta2):
    a1 = Data_Management.add_column_left_of_matrix(a1)
    a2 = g(np.dot(a1, np.transpose(theta1)))      
    a2 = Data_Management.add_column_left_of_matrix(a2)
    
    a3 = g(np.dot(a2, np.transpose(theta2)))
    
    return a1, a2, a3

def backdrop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    """
    return coste y gradiente de una red neuronal de dos capas
    """
    theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)], 
        (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1):], 
        (num_etiquetas, (num_ocultas + 1)))

    #X = Data_Management.add_column_left_of_matrix(X) #columna de 1s
    return 0
X, y = Data_Management.load_mat("ex4data1.mat")

#indexRand = np.random.randint(0, 5001, 100)
#displayData.displayData(X[indexRand[:]])
#plt.show()

weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

a1, a2, a3 = propagation(X, theta1, theta2)
y_transformed = transform_y(y, 10)

print(J(X, y_transformed, a3, 10, theta1, theta2))

print(pesos_aleat(5, 6))
