"""
this file used to start run the first NN
author:bibaodi
"""

import mnist_loader
import network
import network2
import network3
import time


def main_01():
    t1 = time.gmtime(time.time())
    print('first training NN running....%d:%d:%d' % (t1.tm_hour, t1.tm_min, t1.tm_sec))
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    t1 = time.gmtime(time.time())
    print('first training NN running....%d:%d:%d end' % (t1.tm_hour,t1.tm_min, t1.tm_sec))


def main_02():
    t1 = time.gmtime(time.time())
    print('training NN2 running....%d:%d:%d' % (t1.tm_hour, t1.tm_min, t1.tm_sec))
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
    t1 = time.gmtime(time.time())
    print('training NN2 running....%d:%d:%d end' % (t1.tm_hour,t1.tm_min, t1.tm_sec))

def main_03():
    t1 = time.gmtime(time.time())
    print('training NN3 running....%d:%d:%d' % (t1.tm_hour, t1.tm_min, t1.tm_sec))
    from network3 import Network
    from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = network3.Network([
        ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
                  filter_shape = (20, 1, 5, 5),
                  poolsize = (2, 2)),
        ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
                  filter_shape = (40, 20, 5, 5),
                  poolsize = (2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    t1 = time.gmtime(time.time())
    print('training NN3 running....%d:%d:%d end' % (t1.tm_hour,t1.tm_min, t1.tm_sec))

if __name__ == "__main__":
    #main_01()
    #main_02()
    main_03()
