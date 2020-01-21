from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
from ML_UtilsModule import Data_Management, Normalization
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures as pf

NUM_TRIES = 1
feature1 = "attack"
feature2 = "defense"
feature3 = "attack"
grado = 2

def draw_decisition_boundary(X, y, svm, true_score, mu, sigma, c, s, type):
    """
    valid for two feature
    """
    #calculating the graphic
    figure, ax = plt.subplots()
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1], color='blue', marker='o', label = type)
    plt.scatter(X[neg, 0], X[neg, 1], color='black', marker='x', label = "Other types")
    plt.contour(x1, x2, yp, colors=['red', 'purple'])

    #formatting the graphic with some labels
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.suptitle(("Score: " + str(float("{0:.3f}".format(true_score))) + "\n C: " + str(c) + "\n Sigma: " + str(s)))
    figure.legend()

    #set the labels to non-normalized values
    # figure.canvas.draw()
    # labels = [item for item in plt.xticks()[0]]
    # for i in range(len(labels)):
    #     labels[i] = int(round((labels[i] * sigma[0, 0]) + mu[0, 0], -1))
    # ax.xaxis.set_ticklabels(labels)

    # labels = [item for item in plt.yticks()[0]]
    # for i in range(len(labels)):
    #     labels[i] = int(round((labels[i] * sigma[0, 1]) + mu[0, 1], -1))
    # ax.yaxis.set_ticklabels(labels)

    #show
    plt.show()

def draw_3D(X, y, svm, true_score, mu, sigma):
    fig = plt.figure()
    ax = Axes3D(fig)

    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    ax.scatter(X[pos, 0], X[pos, 1], X[pos, 2], color='blue', marker='o', label = "Legendary")
    ax.scatter(X[neg, 0], X[neg, 1], X[neg, 2], color='red', marker='x', label = "Non Legendary")
    plt.suptitle(("Score: " + str(true_score)))
    fig.legend()

    plt.xlabel(feature1)
    plt.ylabel(feature2)

    plt.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    # x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    # x1, x2 = np.meshgrid(x1, x2)
    # yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)

    # ax.plot_surface(x1, x2, yp)

    # pos = (y == 1).ravel()
    # neg = (y == 0).ravel()
    # ax.scatter(X[pos, 0], X[pos, 1], color='blue', marker='o', label = "Legendary")
    # ax.scatter(X[neg, 0], X[neg, 1], color='red', marker='x', label = "Non Legendary")

    # plt.suptitle(("Score: " + str(true_score)))
    # fig.legend()

    # plt.show()

def true_score(X, y, svm):
    predicted_y = svm.predict(X)
    tp = 0
    fp = 0
    fn = 0

    for i in range(np.shape(predicted_y)[0]):
        if predicted_y[i] == 1 and y[i] == 1:
            tp += 1
        elif predicted_y[i] == 1 and y[i] == 0:
            fp += 1
        elif predicted_y[i] == 0 and y[i] == 1:
            fn += 1

    score = 0
    if tp != 0:
        precision_score = tp / (tp + fp)
        recall_score = tp / (tp + fn)
        score = 2 * (precision_score * recall_score) / (precision_score + recall_score)

    return score

def eleccion_parametros_C_y_Sigma(X, y, Xval, yval, mu, sigma):
    possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C_value, sigma = 0, 0
    max_score = float("-inf")
    best_svm = None
    selected_C = 0
    selected_Sigma = 0

    for i in range(len(possible_values)):
        C_value = possible_values[i]
        for j in range(len(possible_values)):
            sigma = possible_values[j]
            svm = SVC(kernel = 'rbf' , C = C_value, gamma= (1 / ( 2 * sigma ** 2)), probability=True)
            svm.fit(X, y.ravel())
            current_score = true_score(Xval, yval, svm) #calcula el score con los ejemplos de validacion (mayor score, mejor es el svm)
            if current_score > max_score:
                max_score = current_score
                best_svm = svm
                selected_C = C_value
                selected_Sigma = sigma

    return best_svm, selected_C, selected_Sigma

def transform_y(y, num_etiquetas):
    y = np.reshape(y, (np.shape(y)[0], 1))
    mask = np.empty((num_etiquetas, np.shape(y)[0]), dtype=bool)
    for i in range( num_etiquetas):
        mask[i, :] = (y[:, 0] == i)

    mask = mask * 1

    return np.transpose(mask)

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

def predict_type(user_values, svms):
    max_security = float("-inf")
    predicted_type = 0

    for i in range(len(svms)):
        sec = svms[i].predict_proba(user_values)
        if sec[0, 1] > max_security:
            max_security = sec[0, 1]
            predicted_type = i

    return max_security, predicted_type


def polynomial_features(X, grado):
    poly = pf(grado)
    return (poly.fit_transform(X))

# X, y = Data_Management.load_csv_types_features("pokemon.csv",['against_bug', 'against_dark','against_dragon','against_electric',
#                          'against_fairy','against_fight','against_fire','against_flying',
#                          'against_ghost','against_grass','against_ground','against_ice','against_normal',
#                          'against_poison','against_psychic','against_rock','against_steel','against_water'])

X, y = Data_Management.load_csv_types_features("pokemon.csv", [feature1, feature2])
# TODO: usar el tipo2 para sacar el score tambien (si mi svm predice 1 y una de las dos y es 1, es truePositive++) y dar el resultado con solo
# 1 tipo, todo lo del entrenamiento se queda igual (se entrena para un solo tipo). Luego en el score se hace eso y para predecir el tipo se queda igual.
# Tambien puedo sacar dos svm, tipo primario y tipo secundario pero mas lio ?


X = polynomial_features(X, grado)
X, mu, sigma = Normalization.normalize_data_matrix(X[:, 1:])
X = Data_Management.add_column_left_of_matrix(X)

trainX, trainY, validationX, validationY, testingX, testingY = divideRandomGroups(X, y)

svms = []

for j in range(18):
    currentTrainY = (trainY == j) * 1
    currentValidationY = (validationY == j) * 1
    currentTestingY = (testingY == j) * 1
    current_svm, C, s = eleccion_parametros_C_y_Sigma(trainX, currentTrainY, validationX, currentValidationY, mu, sigma)
    current_score = true_score(testingX, currentTestingY, current_svm)
    if np.shape(trainX)[1] == 2:
        draw_decisition_boundary(testingX, currentTestingY, current_svm, current_score, mu, sigma, C, s, Data_Management.getTypeByIndex(j))
    svms.append(current_svm)
    print("Score con los ejemplos de testing: " + str(current_score) + " Type: " + Data_Management.getTypeByIndex(j))

while True:
    user_values = np.array(list(map(float, input("Gimme stats: ").split())), dtype=float) # (features, )
    if user_values.size == 0:
        break
    user_values = np.reshape(user_values, (np.shape(user_values)[0], 1))
    user_values = np.transpose(user_values)
    user_values = polynomial_features(user_values, grado)
    user_values = Normalization.normalize(user_values[:, 1:], mu, sigma) #normalization of user values
    sec, pokemon_type = predict_type(user_values, svms)
    print("Predicted type: " + Data_Management.getTypeByIndex(pokemon_type) + ". Probability of that type: " + str(sec) + "\n")
