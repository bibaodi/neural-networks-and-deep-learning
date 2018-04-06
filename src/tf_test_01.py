import tensorflow as tf

def fun_01():
#create the op
    op1=tf.constant([[1.0,2.0],[2.0, 3.0]])
    op2=tf.constant([[0.,1.], [2.0, 0]])
    op3=tf.matmul(op1, op2)

    #create the session and run
    s1 = tf.Session()
    ret = s1.run(op3)
    print(ret)
    s1.close()
    return

def fun_02():
    #create the op
    op1=tf.constant([[1.0,2.0],[2.0, 3.0]])
    op2=tf.constant([[0.,1.], [2.0, 0]])
    op3=tf.matmul(op1, op2)

    #create the session and run
    with tf.Session() as s1:
        ret = s1.run(op3)
        print(ret)
    #s1.close() ##because the 'with sentence' we do not need close session obviously.
    return

#the variables
def fun_variables_01():
    v = tf.Variable(1., name='counter')

    one = tf.constant(1.003)
    value = tf.add(v, one)
    update = tf.assign(v, value)

    #init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()

    with tf.Session() as s:
        s.run(init_op)
        print(s.run(v))#although v is a variable, but the value in variable only by session.run can be updated
        for _ in range(3):
            s.run(update)
            print(s.run(v))

import numpy as np
###a original_full_NN
def tf_DNN_01():
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in xrange(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(W), sess.run(b)

if __name__ == "__main__":
    #fun_01()
    #fun_variables_01()
    tf_DNN_01()