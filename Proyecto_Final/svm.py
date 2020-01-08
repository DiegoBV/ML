from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
from ML_UtilsModule import Data_Management
from sklearn.metrics import confusion_matrix

NUM_TRIES = 1

def draw_decisition_boundary(X, y, svm):
    """
    valid for one feature
    """
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1], color='green', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='pink',
    edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)

def kernel_lineal(X, y):
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

    svm = SVC(kernel='linear', C=100)
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

    return svm

def kernel_gaussiano(X, y):
    sigma = 0.1

    svm = SVC(kernel = 'rbf' , C= 1, gamma= (1 / ( 2 * sigma ** 2)) )
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

    return svm

def true_score(X, y, svm):
    predicted_y = svm.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, predicted_y).ravel()
    score = 0
    if tp != 0:
        precision_score = tp / (tp + fp)
        recall_score = tp / (tp + fn)
        score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    return score

def eleccion_parametros_C_y_Sigma(X, y, Xval, yval):
    possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C_value, sigma = 0, 0
    max_score = 0
    best_svm = None

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
    
    return best_svm

X, y = Data_Management.load_csv_svm("pokemon.csv", ["capture_rate", "base_egg_steps"])
X, y, trainX, trainY, validationX, validationY, testingX, testingY = Data_Management.divide_legendary_groups(X, y)

max_score = float("-inf")
best_svm = None

for i in range(NUM_TRIES):
    #THIS IS GIVING THE SAME RESULT, ALWAYS (MAYBE SELECT C AND SIGMA RANDOMLY)
    seed = np.random.seed()
    current_svm = eleccion_parametros_C_y_Sigma(trainX, trainY, validationX, validationY)
    # plt.figure()
    # draw_decisition_boundary(X, y, current_svm)
    # plt.show()
    current_score = true_score(testingX, testingY, current_svm)
    print("Score con los ejemplos de testing: " + str(current_score))
    if current_score > max_score:
        max_score = current_score
        best_svm = current_svm

while True:
    user_values = np.array(list(map(float, input("Gimme stats: ").split())), dtype=float) # (features, )
    if user_values.size == 0:
        break
    user_values = np.reshape(user_values, (np.shape(user_values)[0], 1))
    user_values = np.transpose(user_values)
    sol = best_svm.predict(user_values)
    print("Is your pokemon legendary?: " + str(sol[0] == 1.0) + "\n")
