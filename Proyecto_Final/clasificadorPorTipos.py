from ML_UtilsModule import Data_Management
from scipy.optimize import minimize as sciMin
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
from matplotlib import pyplot as plt
from ML_UtilsModule import Normalization


lambda_ = 1
NUM_TRIES = 1

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
    y = np.reshape(y, (np.shape(y)[0], 1))
    mask = np.empty((num_etiquetas, np.shape(y)[0]), dtype=bool)
    for i in range( num_etiquetas):
        mask[i, :] = (y[:, 0] == i)

    mask = mask * 1

    return np.transpose(mask)

def des_transform_y(y, num_etiquetas):
    deTransY = np.where(y == 1)
    return deTransY[1]


def divideRandomGroups(X, y):

    X, y = Data_Management.shuffle_in_unison_scary(X, y)
    # ----------------------------------------------------------------------------------------------------
    percent_train = 0.6
    percent_valid = 0.2
    percent_test = 0.2
    # ----------------------------------------------------------------------------------------------------
    # TRAINIG GROUP
    t = int(np.shape(X)[0]*percent_train)
    trainX = X[:t]
    trainY= y[:t]
    # ----------------------------------------------------------------------------------------------------
    # VALIDATION GROUP
    v=int( np.shape(trainX)[0]+np.shape(X)[0]*percent_valid)
    validationX = X[np.shape(trainX)[0] : v]
    validationY= y[np.shape(trainY)[0] : v]
    # ----------------------------------------------------------------------------------------------------
    # TESTING GROUP
    testingX = X[np.shape(trainX)[0]+np.shape(validationX)[0] :]
    testingY= y[np.shape(trainY)[0]+np.shape(validationY)[0] :]

    return trainX, trainY, validationX, validationY, testingX, testingY


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
    maxIndexV = np.argmax(outputLayer, axis = 1)
    checker = (y[:] == maxIndexV)

    truePositives = 0
    falsePositives = 0
    falseNegatives = 0

    for i in range(np.size(checker)):
        if checker[i] == True and y[i] == 1:
            truePositives += 1
        elif checker[i] == True and y[i] == 0:
            falsePositives += 1
        elif checker[i] == False and y[i] == 1:
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

def paint_pkmTypes(X, y, types = None):
    '''
    Creates plt figure and draws the pkms used as input by the first two values of X
    and changes the color of them depending on the type

    X = attributes
    y = list of types, deTransformed (from 0 to 17)
    types = types that are going to be printed
    '''
    typesIndx = []

    if(types == None):
        types = Data_Management.types_
        # esto es para que meta los indices del 0 al 17, de los tipos completos, como no lo conseguia, he pasado
        typesIndx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    else:
        for t in range(len(types)):
            typesIndx.append(Data_Management.types_.index(types[t]))


    colors = Data_Management.colors_

    for i in range(len(typesIndx)):
        pos = (y == typesIndx[i]).ravel()
        plt.scatter(X[pos, 0], X[pos, 1], color=colors[typesIndx[i]], marker='.', label = types[i])


def paint_graphic(X, y, true_score, theta1, theta2, types = None):
    plt.figure()

    paint_pkmTypes(X, y, types)

    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))

    aux = np.c_[ xx1.ravel(), xx2.ravel()]
    
    sigm = forward_propagate( aux, theta1, theta2)[4]
    sigm = np.argmax(sigm, axis = 1) #coge el valor maximo sacado por el frw_Propagate
    sigm = np.reshape(sigm, np.shape(xx1))
    plt.contour(xx1, xx2, sigm, linewidths = 0.25, colors = 'b') 


    plt.xlabel("weight_kg")
    plt.ylabel("speed")
    plt.suptitle(("Score: " + str(true_score)))

    plt.show()

def paint_graphic_norm_full(sigm, xx1, xx2):
    """
    Pinta las funciones de todos los tipos del color de cada tipo
    """
    for i in range(0,18):
        contorn = sigm[:, i]
        contorn = np.reshape(contorn, np.shape(xx1))
        plt.contour(xx1, xx2, contorn, linewidths = 0.5, colors=Data_Management.colors_[i])
        
def paint_graphic_norm_partial(sigm, xx1, xx2, types = None):
    """
    Pinta las funciones de los tipos elegidos del color de cada tipo
    """
    typesIndx = []
    for t in range(len(types)):
        typesIndx.append(Data_Management.types_.index(types[t]))
        
    for i in range(len(typesIndx)):
        contorn = sigm[:, typesIndx[i]]
        contorn = np.reshape(contorn, np.shape(xx1))
        plt.contour(xx1, xx2, contorn, linewidths = 0.5, colors=Data_Management.colors_[typesIndx[i]])
    
    
def paint_graphic_norm(X, y, true_score, theta1, theta2, mu, sigma, graphic_attr_names, types = None, show_allTypes = False):     
    figure, ax = plt.subplots()
    
    if show_allTypes:
        paint_pkmTypes(X, y)
    else:
        paint_pkmTypes(X, y, types)
    
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))
    
    aux = np.c_[ xx1.ravel(), xx2.ravel()]
    #p, aux = polinomial_features(aux, 2)
    #aux = Normalization.normalize(aux[:, 1:], mu, sigma)
    sigm = forward_propagate(aux, theta1, theta2)[4]
    
    if types == None:
        paint_graphic_norm_full(sigm, xx1, xx2)
    else:
        paint_graphic_norm_partial(sigm, xx1, xx2, types)

    #formatting the graphic with some labels
    plt.xlabel(graphic_attr_names[0])
    plt.ylabel(graphic_attr_names[1])
    plt.suptitle(("Score: " + str(float("{0:.3f}".format(true_score)))))
    figure.legend()

    #set the labels to non-normalized values
    figure.canvas.draw()
    labels = [item for item in plt.xticks()[0]]
    for i in range(len(labels)):
        labels[i] = int(round((labels[i] * sigma[0, 0]) + mu[0, 0], -1))
    ax.xaxis.set_ticklabels(labels)

    labels = [item for item in plt.yticks()[0]]
    for i in range(len(labels)):
        labels[i] = int(round((labels[i] * sigma[0, 1]) + mu[0, 1], -1))
    ax.yaxis.set_ticklabels(labels)
    
    plt.show()


attr_names = ["capture_rate", "base_egg_steps"]
types_to_paint =["fire"]
X, y = Data_Management.load_csv_types_features("pokemon.csv", attr_names)



#normalize
X, mu, sigma = Normalization.normalize_data_matrix(X)



num_entradas = np.shape(X)[1]
num_ocultas = 25
num_etiquetas = 18
true_score_max = float("-inf")

thetaTrueMin1 = None
thetaTrueMin2 = None

y_transformed =  transform_y(y, num_etiquetas)

trainX, trainY, validationX, validationY, testingX, testingY = divideRandomGroups(X, y_transformed)

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


    true_score = checkLearned(y, forward_propagate(X, thetaMin1, thetaMin2)[4])
    if true_score > true_score_max:
        true_score_max = true_score
        thetaTrueMin1 = thetaMin1
        thetaTrueMin2 = thetaMin2
        pintaTodo(testingX, testingY, auxErr, auxErrTr, true_score)

deTransTestY = des_transform_y(testingY, num_etiquetas);
# Pintado ---------------------------------------------------------------------------------------------------------------------------
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["bug", "normal", 'flying', 'ghost'], True)
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["bug", "normal", 'flying', 'ghost'])

paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["bug"])
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["normal"])
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["flying"])
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["ghost"])

paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["bug"], True)
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["normal"], True)
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["flying"], True)
paint_graphic_norm(testingX, deTransTestY, true_score_max, thetaTrueMin1, thetaTrueMin2, mu, sigma, attr_names, ["ghost"], True)
# -----------------------------------------------------------------------------------------------------------------------------------

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
