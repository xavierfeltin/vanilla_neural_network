import mnist_loader
import network2
import network

def call_network1():
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def call_network2():
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5,
            lmbda=5.0,
            evaluation_data=validation_data,
            L1_regularization=True,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    call_network2()