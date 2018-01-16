"""multiple_regularization
~~~~~~~~~~~~~~~
This program shows how different lambda coefficient on L2 weight decay affect training.
In particular, we'll plot out how the cost changes.
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
LMBDAS = [0.5,5,50]

COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30

def main():
    #run_networks()
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
    for lmbda in LMBDAS:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        print ("\nTrain a network using lmbda = "+str(lmbda))
        net = network2.Network([784, 30, 10])
        results.append(
            net.SGD(training_data, NUM_EPOCHS, 10, LEARNING_RATE, lmbda=lmbda,
                    p_evaluation_data=validation_data,
                    monitor_training_cost=True))

    f = open("multiple_decays.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_decays.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lmbda, result, color in zip(LMBDAS, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="$\lambda$ = " + str(lmbda),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
