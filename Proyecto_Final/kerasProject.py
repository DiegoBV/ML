import os
import numpy as np
np.random.seed(0)
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import metrics
from ML_UtilsModule import Data_Management

epoch_count=0
count=0

def plot_decision_boundary(X, y, model,epoch_count,count,steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
    fig.suptitle("Epoch: "+str(epoch_count), fontsize=10)
    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    plt.show()
    return epoch_count

def plot_loss_accuracy(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Keras callback to save decision boundary after each epoch
class prediction_history(Callback):
    def __init__(self):
        self.epoch_count=epoch_count
        self.count=count
    def on_epoch_end(self,epoch,logs={}):
        if self.epoch_count%100==0 or self.epoch_count==399:
            plot_decision_boundary(X, y, model,self.epoch_count,self.count,cmap='RdBu')
            score, acc = model.evaluate(X, y, verbose=0)
            print("error " + str(score) + ", accuracy " + str(acc))
            self.count=self.count+1
        self.epoch_count=self.epoch_count+1
        return self.epoch_count



if __name__ == "__main__":
    X, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)
    X, y = Data_Management.load_csv_svm("pokemon.csv", ['weight_kg', 'height_m'])
    
    X, y, trainX, trainY, validationX, validationY, testingX, testingY = Data_Management.divide_legendary_groups(X, y)
    
    y  =  y.ravel()
    trainY = trainY.ravel()
    # Create a directory where image will be saved
    os.makedirs("images_new", exist_ok=True)

    # Define our model object
    model = Sequential()
    # Add layers to our model
    model.add(Dense(units=25, input_shape=(2,), activation="sigmoid", kernel_initializer="glorot_uniform"))
    model.add(Dense(units=25, activation="sigmoid", kernel_initializer="glorot_uniform"))
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer="glorot_uniform"))
    sgd = SGD(lr=0.1)
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    predictions=prediction_history()
    history = model.fit(X, y, validation_split = 0.1, verbose=0,epochs=400, shuffle=True,callbacks=[predictions])
    
    plot_loss_accuracy(history)