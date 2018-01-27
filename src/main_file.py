"""
this file used to start run the first NN
author:bibaodi
"""

import mnist_loader
import network
import network2
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


if __name__ == "__main__":
   main_01()
   # main_02()
