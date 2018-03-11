"""
this file is uesd to learn numpy
author:bibaodi
date:20180109
"""
import numpy

def t1():
    ret1 = numpy.random.randn(1)
    ret11 = numpy.random.randn(1,1)
    ret12 = numpy.random.randn(1,2)
    ret21 = numpy.random.randn(2,1)
    ret22 = numpy.random.randn(2,2)
    ret111 = numpy.random.randn(1,1,1)
    ret = numpy.random.randn(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    print("1,=%s, 1,1=%s, 1,2=%s, 2,1=%s, 2,2=%s 1,1,1=%s alot1=%s" %(ret1,ret11,ret12,ret21,ret22,ret111,ret))

    ret3d = numpy.random.randn(2,2,2)
    print("2,2,2=%s" %(ret3d))
    ret4d = numpy.random.randn(2,2,2,1)
    print("4d=%s" %(ret4d))
    return

def t2():
    sizes = [2,3,1,4]
    print(sizes[1:])
    return

def t3():
    sizes = [2,3,1,4]
    print(sizes[:-1])
    print(zip(sizes[:-1],sizes[1:]))
    return

def t4():
    sizes = [2,3,1,4]
    #print(sizes[:-1])
    biases = [numpy.random.randn(x,1) for x in sizes[1:]]
    weights = [numpy.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
    #zip(sizes[:-1],sizes[1:])
    print("biases:%s\n wieght%s" % (biases, weights))
    return

def iterate():
    sizes = [2,3,1,4]
    for x in sizes:
        print(x)
    return

def fun_exp(x):
    y = numpy.exp(x)
    print("exp(x)=%s" %y)
    return y

def fun_zip():
    a = [2,3,1]
    b = [3,2,6,7,9]
    for b,w in zip(a,b):
        print("(%s,%s)" %(b,w))
    sizes = [2,3,1,4]
    biases = [numpy.random.randn(x,1) for x in sizes[1:]]
    weights = [numpy.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
    for b,w in zip(biases, weights):
        print("(%s,%s)" %(b,w))
    return b

if __name__ == '__main__':
    print('starting....\n')
    t1()
    print('==============\n')
    t2()
    print('==============\n')
    t3()
    print('==============\n')
    t4()
    print('==============\n')
    iterate()
    print('==============\n')
    fun_exp(0)
    fun_exp(1)
    print('==============\n')
    fun_zip()
    print('...end\n')