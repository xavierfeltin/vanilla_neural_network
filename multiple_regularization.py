"""multiple_regularization
~~~~~~~~~~~~~~~
This program shows how different regularization affect training.
In particular, we'll plot out how the cost changes using L2 and L1 regularization.

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATE = 0.025
COLORS = ['#2A6EA6', '#FFCD33']
NUM_EPOCHS = 30

def main():
    run_networks()
    make_plot()

def run_networks():
    """Train networks using three different values for the learning rate,
    and store the cost curves in the file ``multiple_eta.json``, where
    they can later be used by ``make_plot``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    results = []

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print ("\nTrain a network using L2")
    net = network2.Network([784, 30, 10])
    results.append(
        net.SGD(training_data, NUM_EPOCHS, 10, LEARNING_RATE, lmbda=5.0,
                p_evaluation_data=validation_data,
                monitor_training_cost=True))

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("\nTrain a network using L1")
    net = network2.Network([784, 30, 10])
    results.append(
        net.SGD(training_data, NUM_EPOCHS, 10, LEARNING_RATE, lmbda=5.0,
                p_evaluation_data=validation_data,
                L1_regularization=True,
                monitor_training_cost=True))

    f = open("multiple_regularization.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_regularization.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for regul, result, color in zip(['L2', 'L1'], results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="Regul " + regul,
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
