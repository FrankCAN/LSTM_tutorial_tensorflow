import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   # image shape:28 * 28
n_steps = 28    # time steps
n_hidden_units = 256
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden_units*2, n_classes]))

biases = tf.Variable(tf.random_normal([n_classes]))


def BiRNN(X, weights, biases):
    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, n_inputs])
    X = tf.split(X, n_steps)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X, dtype=tf.float32)



    # hidden layer for outputs as the final results
    results = tf.matmul(outputs[-1], weights) + biases

    return results


pred = BiRNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))

        step += 1



