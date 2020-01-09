import numpy as np
import csv as csv
from pandas.io.parsers import read_csv
from scipy.io import loadmat
from matplotlib import pyplot as plt
               

class Data_Management:

    @staticmethod
    def load_csv(file_name):
        """
        Load the csv file. Returns numpy array
        """
        dataFile = read_csv(file_name, header = 0)

        #dataFile['Gender'] = dataFile['Sex'].map({'female':0, 'male':1, }).astype(int)
        #dataMatrix['pokedex_number'] = dataFile['pokedex_number']
        #dataFile = dataFile.drop(["Minior"], axis = 775) #todo dropear esta fila no sabemos como
        dataFile = dataFile.fillna(0)
        y = dataFile['is_legendary'].array

        dataFile = dataFile.drop(['abilities', 'classfication', 'japanese_name', 'name', 'type1', 'type2', 
                                  'against_bug', 'against_dark','against_dragon','against_electric','against_fairy','against_fight','against_fire','against_flying','against_ghost','against_grass','against_ground','against_ice','against_normal','against_poison','against_psychic','against_rock','against_steel','against_water',
                                  'is_legendary'], axis =1).values
    
        #dataFile = dataFile['capture_rate'].array
        
        return dataFile, y
    
    @staticmethod
    def load_csv_RedNeuronalV01(file_name):
        """
        Load the csv file. Returns numpy array
        """
        dataFile = read_csv(file_name, header = 0)

        #dataFile['Gender'] = dataFile['Sex'].map({'female':0, 'male':1, }).astype(int)
        #dataMatrix['pokedex_number'] = dataFile['pokedex_number']
        #dataFile = dataFile.drop(["Minior"], axis = 775) #todo dropear esta fila no sabemos como
        dataFile = dataFile.fillna(0)
        y = dataFile['is_legendary'].array

        dataFile = dataFile.drop(['abilities', 'classfication', 'japanese_name', 'name', 'type1', 'type2', 
                                  'against_bug', 'against_dark','against_dragon','against_electric','against_fairy','against_fight','against_fire','against_flying','against_ghost','against_grass','against_ground','against_ice','against_normal','against_poison','against_psychic','against_rock','against_steel','against_water',
                                  'is_legendary'], axis =1).values
        return dataFile, y

    @staticmethod
    def load_csv_svm(file_name, features):
        """
        Load the csv file. Returns numpy array
        """
        dataFile = read_csv(file_name, header = 0)

        dataFile = dataFile.fillna(0)
        y = dataFile['is_legendary'].array

        X = np.array([])
        X = np.reshape(X, (len(y), 0))
        for i in range(len(features)):
            X = np.c_[X, dataFile[features[i]].array]
        
        return X, y

    @staticmethod
    def add_column_left_of_matrix(matrix):
        new_matrix = np.hstack([np.ones([np.shape(matrix)[0], 1]), matrix]) #convention in linear regr
        return new_matrix

    @staticmethod
    def add_column_top_of_matrix(matrix):
        new_matrix = np.vstack([np.ones([1, np.shape(matrix)[1]]), matrix]) #convention in linear regr
        return new_matrix

    @staticmethod
    def shuffle_in_unison_scary(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

        return a, b

    @staticmethod
    def divide_legendary_groups(X, y):
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (np.shape(y)[0], 1))

        X, y = Data_Management.shuffle_in_unison_scary(X, y)
        # ----------------------------------------------------------------------------------------------------
        legendPos = np.where(y == 1)
        legendX = X[legendPos[0]]
        legendY = y[legendPos[0]]

        normiePos = np.where(y == 0)
        normieX = X[normiePos[0]]
        normieY = y[normiePos[0]]
        # ----------------------------------------------------------------------------------------------------
        # TRAINIG GROUP
        normTrain = int(np.shape(normieX)[0]/4)
        trainX = normieX[:normTrain]
        trainY= normieY[:normTrain]

        legendTrain = int(np.shape(legendX)[0]/2)
        trainX = np.concatenate((trainX, legendX[:legendTrain]))
        trainY = np.concatenate((trainY, legendY[:legendTrain]))
        trainX, trainY = Data_Management.shuffle_in_unison_scary(trainX, trainY)
        # ----------------------------------------------------------------------------------------------------
        # VALIDATION GROUP
        normValid = int(np.shape(normieX)[0]/2)
        validationX = normieX[normTrain:normValid+normTrain]
        validationY = normieY[normTrain:normValid+normTrain]

        legendValid = int(np.shape(legendX)[0]/4)
        validationX = np.concatenate((validationX, legendX[legendTrain:legendValid+legendTrain]))
        validationY = np.concatenate((validationY, legendY[legendTrain:legendValid+legendTrain]))
        validationX, validationY = Data_Management.shuffle_in_unison_scary(validationX, validationY)
        # ----------------------------------------------------------------------------------------------------
        # TESTING GROUP
        testingX = normieX[normValid+normTrain:]
        testingX = np.concatenate((testingX, legendX[legendValid+legendTrain:]))

        testingY = normieY[normValid+normTrain:]
        testingY = np.concatenate((testingY, legendY[legendValid+legendTrain:]))
        testingX, testingY = Data_Management.shuffle_in_unison_scary(testingX, testingY)

        return X, y, trainX, trainY, validationX, validationY, testingX, testingY

class Normalization:
    """
    Regression Normalization (adding the one columns)
    """
    
    @staticmethod
    def normalize_data_matrix(X):
        """
        Normalize the given matrix using the mean and deviation of every column (attribute).
        Returns the normalized matrix, the mu vector (mean of every attribute) and sigma vector (deviation of every attribute).
        """
        n = np.shape(X)[1]
        X_norm = np.empty_like(X)
        sigma = np.empty(n)
        mu = np.empty(n)

        for i in range(0, n):
            mu[i] = np.mean(X[:, i]) #mean of every column or attribute
            sigma[i] = np.std(X[:, i]) #deviation of every column or attribute

            new_value = (X[:, i] - mu[i])/sigma[i] #normalize
            X_norm[:, i] = new_value

        mu = np.reshape(mu, (1, np.shape(mu)[0]))
        sigma = np.reshape(sigma, (1, np.shape(sigma)[0]))

        return X_norm, mu, sigma

    @staticmethod
    def normalize(X, mu, sigma):
        """
        Normalize a matrix using mu and sigma vectors obtained in a
        previous normalization 
        """
        aux1 = X[:, :] - mu[0, :]
        return aux1[:, :] / sigma[0, :]

    @staticmethod
    def normalize_single_attributes(attributes, mu, sigma):
        """
        Normalize a single set of different attributes using mu and sigma vectors obtained in a
        previous normalization 
        """
        new_data = (attributes - mu[:])/sigma[:] #normalize
        return new_data