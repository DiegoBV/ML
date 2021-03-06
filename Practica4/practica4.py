from ML_UtilsModule import Data_Management
from scipy.optimize import minimize as sciMin
from scipy.io import loadmat
from matplotlib import pyplot as plt
import displayData
import numpy as np
import checkNNGradients as check

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

    #--------------------PASO1---------------------------------------
    a1, a2, a3 = propagation(X, theta1, theta2)
    m = np.shape(X)[0]
    #--------------------PASO2---------------------------------------
    #delta_3 = a3 - y # (5000, 10)
    delta_matrix_1 = np.zeros(np.shape(theta1))
    delta_matrix_2 = np.zeros(np.shape(theta2))
    
    for i in range(np.shape(X)[0]):
        delta3 = a3[i] - y[i] #(3, )
        aux1 = np.dot(np.transpose(theta2), delta3[:, np.newaxis]) #(6, 1)
        aux2 = derivada_de_G(np.dot(theta1, a1[i][:, np.newaxis])) #(5, 1)
        aux1 = np.delete(aux1, [0], axis=0) #(5, 1) linea de la cual no estamos nada seguros ??
        delta2 = aux1 * aux2 #(5, 1)
        delta_matrix_1 = delta_matrix_1 + np.dot(delta2, np.transpose(a1[i][:, np.newaxis])) #(5, 4)
        delta_matrix_2 = delta_matrix_2 + np.dot(delta3[:, np.newaxis], np.transpose(a2[i][:, np.newaxis])) #(3, 6)


    #--------------------PASO3---------------------------------------

    # for t in range(np.shape(X)[0]):
    #     a1t = a1[t, :]  # (1, 401)
    #     a2t = a2[t, :]  # (1, 26)
    #     ht = a3[t, :]  # (1, 10)
    #     yt = y[t]  # (1, 10)

    #     d3t = ht - yt  # (1, 10)
    #     d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))  # (1, 26)

    #     delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
    #     delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])


    # aux1 = np.dot(delta_3, theta2) #(5000, 26)
    # aux2 = Data_Management.add_column_left_of_matrix(derivada_de_G(np.dot(a1, np.transpose(theta1))))
    # delta_2 = aux1 * aux2 #(5000, 26)
    # delta_2 = np.delete(delta_2, [0], axis=1) #(5000, 25)

    # #--------------------PASO4---------------------------------------
    # delta_matrix_1 = np.zeros(np.shape(theta1))
    # delta_matrix_2 = np.zeros(np.shape(theta2))

    # delta_matrix_1 = delta_matrix_1 + np.transpose(np.dot(np.transpose(a1), delta_2)) #(25, 401)
    # delta_matrix_2 = delta_matrix_2 + np.transpose(np.dot(np.transpose(a2), delta_3)) #(10, 26)
   
    # #--------------------PASO5---------------------------------------
    # delta_matrix_1 = delta_matrix_1/np.shape(X)[0]
    # delta_matrix_2 = delta_matrix_2/np.shape(X)[0]

    # #--------------------PASO6---------------------------------------
    delta_matrix_1 = (1/m) * delta_matrix_1
    delta_matrix_1[:, 1:] = delta_matrix_1[:, 1:] + (reg/m) * theta1[:, 1:] 

    delta_matrix_2 = (1/m) * delta_matrix_2
    delta_matrix_2[:, 1:] = delta_matrix_2[:, 1:] + (reg/m) * theta2[:, 1:] 
    
    
    cost = J(X, y, a3, num_etiquetas, theta1, theta2)
    gradient = np.concatenate((np.ravel(delta_matrix_1), np.ravel(delta_matrix_2)))
    
    return cost, gradient


X, y = Data_Management.load_mat("ex4data1.mat")

#indexRand = np.random.randint(0, 5001, 100)
#displayData.displayData(X[indexRand[:]])
#plt.show()

weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
theta1 = pesos_aleat(np.shape(theta1)[1]-1,np.shape(theta1)[0])
theta2 = pesos_aleat(np.shape(theta2)[1]-1,np.shape(theta2)[0])


#a1, a2, a3 = propagation(X, theta1, theta2)

theta_vector = np.concatenate((np.ravel(theta1), np.ravel(theta2)))


fmin = sciMin(fun=backdrop, x0=theta_vector,
 args=(np.shape(X)[1], 25, 10, X, y, learning_rate),
 method='TNC', jac=True,
 options={'maxiter': 70})

#backdrop(theta_vector, np.shape(X)[1], 25, 10, X, y, learning_rate)
print(check.checkNNGradients(backdrop, 1e-4))
#print(J(X, transform_y(y, 10), a3, 10, theta1, theta2))

#print(pesos_aleat(5, 6))
