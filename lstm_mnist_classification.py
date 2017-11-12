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

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # (128)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # (10)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X (128, 28 steps, 28 inputs) ===> (128 * 28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in (128 * 28, 128)
    X_in = tf.matmul(X, weights['in'])  + biases['in']
    # X_in (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # cell
    # lstm cell is divided into two parts (c_state, m_state)
    # c_state means last cell state, m_state means output state
    # 其中c代表Ct的最后时间的输出，h代表Ht最后时间的输出，h是等于最后一个时间的output的
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)


    # Method 2 for cell calculation
    # outputs = []
    # states = _init_state
    # with tf.variable_scope("RNN"):
    #     for time_step in range(n_steps):
    #         if time_step > 0:
    #             tf.get_variable_scope().reuse_variables()  # LSTM同一曾参数共享，
    #         (cell_out, states) = lstm_cell(X_in[:, time_step, :], states)
    #         outputs.append(cell_out)


    # hidden layer for outputs as the final results
    results = tf.matmul(states[1], weights['out']) + biases['out']

    # or unpack to list [(batch, outputs)...] * steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
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



