import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# KERAS THINGS
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
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

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

    return fig, ax

X, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)
colors = ['blue' if label == 1 else 'red' for label in y]
plt.scatter(X[:,0], X[:,1], color=colors)

# Define our model object
model = Sequential()

# kwarg dict for convenience
layer_kw = dict(activation='sigmoid', init='glorot_uniform')

# Add layers to our model
model.add(Dense(output_dim=5, input_shape=(2, ), **layer_kw))
model.add(Dense(output_dim=5, **layer_kw))
model.add(Dense(output_dim=1, **layer_kw))

# Define the Optimizer
sgd = SGD(lr=0.001)

model.compile(optimizer=sgd, loss='binary_crossentropy')

# Train
history = model.fit(X[:500], y[:500], verbose=0, nb_epoch=4000, shuffle=True)

# Paint
plot_decision_boundary(X, y, model, cmap='RdBu')