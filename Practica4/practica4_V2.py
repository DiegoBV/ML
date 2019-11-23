from ML_UtilsModule import Data_Management
from scipy.optimize import minimize as sciMin
from scipy.io import loadmat
import numpy as np
import displayData
from matplotlib import pyplot as plt
import checkNNGradients as check

lambda_ = 1

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
    #y = np.reshape(y, (np.shape(y)[0], 1))
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
    return (1/m) * np.sum(aux3) + (lambda_/(2*m)) * aux4

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), g(z2)])
    z3 = np.dot(a2, theta2.T)
    h = g(z3)
    return a1, z2, a2, z3, h

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

    #--------------------PASO1---------------------------------------
   
    a1, a2, a3 = propagation(X, theta1, theta2)
    m = np.shape(X)[0]
    delta_3 = a3 - y # (5000, 10)
    #--------------------PASO2---------------------------------------
    #delta_3 = a3 - y # (5000, 10)
    delta_matrix_1 = np.zeros(np.shape(theta1))
    delta_matrix_2 = np.zeros(np.shape(theta2))
    
    aux1 = np.dot(delta_3, theta2) #(5000, 26)
    aux2 = Data_Management.add_column_left_of_matrix(derivada_de_G(np.dot(a1, np.transpose(theta1))))
    delta_2 = aux1 * aux2 #(5000, 26)
    delta_2 = np.delete(delta_2, [0], axis=1) #(5000, 25)

    # #--------------------PASO4---------------------------------------

    delta_matrix_1 = delta_matrix_1 + np.transpose(np.dot(np.transpose(a1), delta_2)) #(25, 401)
    delta_matrix_2 = delta_matrix_2 + np.transpose(np.dot(np.transpose(a2), delta_3)) #(10, 26)
    #--------------------PASO6---------------------------------------
    delta_matrix_1 = (1/m) * delta_matrix_1
    delta_matrix_1[:, 1:] = delta_matrix_1[:, 1:] + (reg/m) * theta1[:, 1:] 

    delta_matrix_2 = (1/m) * delta_matrix_2
    delta_matrix_2[:, 1:] = delta_matrix_2[:, 1:] + (reg/m) * theta2[:, 1:] 
    
    
    cost = J(X, y, a3, num_etiquetas, theta1, theta2)
    gradient = np.concatenate((np.ravel(delta_matrix_1), np.ravel(delta_matrix_2)))
    
    return cost, gradient

def checkLearned(y, outputLayer):
    
    maxIndexV = np.argmax(outputLayer, axis = 1) + 1
    
    checker = (y[:,0] == maxIndexV) 
    count = np.size(np.where(checker == True)) 
    fin = count/np.shape(y)[0] * 100
    return fin


X, y = Data_Management.load_mat("ex4data1.mat")

indexRand = np.random.randint(0, 5001, 100)
displayData.displayData(X[indexRand[:]])
plt.show()


weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
theta1 = pesos_aleat(np.shape(theta1)[1]-1,np.shape(theta1)[0])
theta2 = pesos_aleat(np.shape(theta2)[1]-1,np.shape(theta2)[0])
num_entradas = np.shape(X)[1]
num_ocultas = 25
num_etiquetas = 10

theta_vector = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

thetas = sciMin(fun=backdrop, x0=theta_vector,
 args=(num_entradas, num_ocultas, num_etiquetas, X, transform_y(y, num_etiquetas), lambda_),
 method='TNC', jac=True,
 options={'maxiter': 70}).x

theta1 = np.reshape(thetas[:num_ocultas*(num_entradas + 1)], 
        (num_ocultas, (num_entradas + 1)))
theta2 = np.reshape(thetas[num_ocultas*(num_entradas + 1):], 
        (num_etiquetas, (num_ocultas + 1)))

print("Precision de la red neuronal: " + str(checkLearned(y, forward_propagate(X, theta1, theta2)[4])) + " %")
    
    

#print(check.checkNNGradients(backdrop, 0))
