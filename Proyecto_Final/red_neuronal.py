from ML_UtilsModule import Data_Management
from scipy.optimize import minimize as sciMin
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
from matplotlib import pyplot as plt
from ML_UtilsModule import Normalization


lambda_ = 1
NUM_TRIES = 3

def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))

def derivada_de_G(z):
    result = g(z) * (1 - g(z))
    return result

def polinomial_features(X, grado):
    poly = pf(grado)
    return (poly, poly.fit_transform(X))

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
    checker = (outputLayer > 0.7)
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    
    for i in range(np.size(checker)):
        if checker[i, 0] == True and y[i, 0] == 1:
            truePositives += 1
        elif checker[i, 0] == True and y[i, 0] == 0:
            falsePositives += 1
        elif checker[i, 0] == False and y[i, 0] == 1:
            falseNegatives += 1    
    
    if truePositives == 0:
        return 0
        
    recall = (truePositives/(truePositives + falseNegatives)) 
    precision = (truePositives/(truePositives + falsePositives))
    score = 2 *(precision*recall/(precision + recall))
    
    # PORCENTAJE DE ACIERTOS TOTALES
    #count = np.size(np.where(checker[:, 0] == çy[:, 0])) 
    #fin = count/np.shape(y)[0] * 100
    
    return score 

def pintaTodo(X, y, error, errorTr, true_score):
    plt.figure()
    #plt.ylim(0,1)
    plt.plot(np.linspace(0,len(error)-1, len(error), dtype = int), error[:], color="grey")
    plt.plot(np.linspace(0,len(errorTr)-1, len(errorTr), dtype = int), errorTr[:], color="green")
    plt.suptitle(("Score: " + str(true_score)))
    
    plt.show()
    
def paint_graphic(X, y, true_score, theta1, theta2):     
    plt.figure()
    
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1], color='blue', marker='o', label = "Legendary")
    plt.scatter(X[neg, 0], X[neg, 1], color='black', marker='x', label = "Non legendary")   
    
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))
    
    sigm = forward_propagate(np.c_[ xx1.ravel(), xx2.ravel()], theta1, theta2)[4]
    sigm = np.reshape(sigm, np.shape(xx1))
    plt.contour(xx1, xx2, sigm, [0.5], linewidths = 1, colors = 'g')
    
    plt.suptitle(("Score: " + str(true_score)))
    
    plt.show()
        
        
X, y = Data_Management.load_csv_svm("pokemon.csv", ["base_total", "base_happiness"])

#normalize
#X, mu, sigma = Normalization.normalize_data_matrix(X)

X, y, trainX, trainY, validationX, validationY, testingX, testingY = Data_Management.divide_legendary_groups(X, y)


num_entradas = np.shape(X)[1]
num_ocultas = 25
num_etiquetas = 1
true_score_max = float("-inf")

thetaTrueMin1 = None
thetaTrueMin2 = None

for j in range(NUM_TRIES):
    theta1 = pesos_aleat(num_entradas, num_ocultas)
    theta2 = pesos_aleat(num_ocultas, num_etiquetas)

    theta_vector = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

    auxErr = []
    auxErrTr = []
    thetaMin1 = None
    thetaMin2 = None
    errorMin = float("inf")

    for i in range(1, np.shape(trainX)[0]):
        thetas = sciMin(fun=backdrop, x0=theta_vector,
        args=(num_entradas, num_ocultas, num_etiquetas, trainX[:i], trainY[:i], lambda_),
        method='TNC', jac=True,
        options={'maxiter': 70}).x

        theta1 = np.reshape(thetas[:num_ocultas*(num_entradas + 1)], 
                (num_ocultas, (num_entradas + 1)))
        theta2 = np.reshape(thetas[num_ocultas*(num_entradas + 1):], 
                (num_etiquetas, (num_ocultas + 1)))
        
        auxErr.append(J(validationX, validationY, forward_propagate(validationX, theta1, theta2)[4], num_etiquetas, theta1, theta2))
        auxErrTr.append(J(trainX[:i], trainY[:i], forward_propagate(trainX[:i], theta1, theta2)[4], num_etiquetas, theta1, theta2))
        
        if errorMin > auxErr[-1]:
            errorMin = auxErr[-1]
            thetaMin1 = theta1
            thetaMin2 = theta2


    true_score = checkLearned(testingY, forward_propagate(testingX, thetaMin1, thetaMin2)[4])
    if true_score > true_score_max:
        true_score_max = true_score
        thetaTrueMin1 = thetaMin1
        thetaTrueMin2 = thetaMin2
        pintaTodo(testingX, testingY, auxErr, auxErrTr, true_score)

paint_graphic(testingX, testingY, true_score_max, thetaTrueMin1, thetaTrueMin2);

print("True Score de la red neuronal: " + str(true_score_max) + "\n")
while True:
    user_values = np.array(list(map(float, input("Gimme stats: ").split())), dtype=float) # (features, )
    if user_values.size == 0:
        break
    user_values = np.reshape(user_values, (np.shape(user_values)[0], 1))
    user_values = np.transpose(user_values)
    user_values = Normalization.normalize(user_values, mu, sigma) #normalization of user values
    sol = forward_propagate(user_values, thetaTrueMin1, thetaTrueMin2)[4]
    print("Is your pokemon legendary?: " + str(sol[0, 0] > 0.7) + "\n")

#print(check.checkNNGradients(backdrop, 0))
