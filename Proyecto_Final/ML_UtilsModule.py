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
    def get_parameters_from_training_examples(training_examples):
        """
        Returns the needed parameters from the training examples, aka X, Y, m, n
        """
        X = training_examples[:, :-1] #every col but the last
        m = np.shape(X)[0] #number of training examples
        n = np.shape(X)[1] #number of attributes
        Y = training_examples[:, -1] #the last col, every row
        Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is (needed for numpy operations)

        return X, Y, m, n

    @staticmethod
    def add_column_left_of_matrix(matrix):
        new_matrix = np.hstack([np.ones([np.shape(matrix)[0], 1]), matrix]) #convention in linear regr
        return new_matrix

    @staticmethod
    def add_column_top_of_matrix(matrix):
        new_matrix = np.vstack([np.ones([1, np.shape(matrix)[1]]), matrix]) #convention in linear regr
        return new_matrix
    
    @staticmethod
    def load_mat(file_name):
        data = loadmat(file_name)
        y = data['y']
        X = data['X']
        return X, y

    @staticmethod
    def draw_random_examples(X):
        sample = np.random.choice(X.shape[0], 10)
        plt.imshow(X[sample, :].reshape(-1, 20).T)
        plt.axis('off')


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

        return X_norm, mu, sigma

    @staticmethod
    def normalize(X, mu, sigma):
        """
        Normalize the given matrix using the mean and deviation of every column (attribute).
        Returns the normalized matrix, the mu vector (mean of every attribute) and sigma vector (deviation of every attribute).
        """
        n = np.shape(X)[1]
        X_norm = np.empty_like(X)

        for i in range(0, n):
            new_value = (X[:, i] - mu[i])/sigma[i] #normalize
            X_norm[:, i] = new_value

        return X_norm

    @staticmethod
    def normalize2(X, mu, sigma):
        aux1 = X[:, :] - mu[0, :]
        return aux1[:, :]/sigma[0, :]

    @staticmethod
    def normalize_single_attributes(attributes, mu, sigma):
        """
        Normalize a single set of different attributes using mu and sigma vectors obtained in a
        previous normalization 
        """
        new_data = (attributes - mu[:])/sigma[:] #normalize
        return new_data