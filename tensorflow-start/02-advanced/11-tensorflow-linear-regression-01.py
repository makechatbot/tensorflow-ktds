import tensorflow as tf

x_data = [1.,2.,3.,4.]
y_data = [3.,6.,9.,12.]

W = tf.Variable(tf.random_uniform([1], -10., 10.))
b = tf.Variable(tf.random_uniform([1], -10., 10.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

rate = tf.Variable(0.1)

optimizer = tf.train.GradientDescentOptimizer(rate)

train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
