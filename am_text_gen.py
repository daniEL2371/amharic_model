import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from am_process_text import DataGen
from tensorflow.contrib import rnn

seq_length = 100
batch_size = 200
hidden_size = 64

print("Generating Data")
gen = DataGen("data/small.txt", "data/geez.txt", batch_size, seq_length)
gen.process()
print("Total Training Data: {0}".format(len(gen.train_X)))
x_dims, y_dims, z_dims = gen.data_dims

epoches = 20
batches = len(gen.train_X) // batch_size
iterations = epoches * batches
print("Batches: {0}, Epoches: {1}".format(batches, epoches))


def RNN(x, weight, bias, name):
    xx = tf.unstack(x, seq_length, 1)
    rnn_cell = rnn.MultiRNNCell([
        rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, name=name + str(0)),
        rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, name=name + str(1))
    ])
    outputs, states = rnn.static_rnn(rnn_cell, xx, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias


print("Setting up input and variables")
with tf.name_scope("Input") as scope:
    in_x = tf.placeholder(
        tf.float32, [None, seq_length, 1], name="X_context_chars")
    in_y = tf.placeholder(tf.int32, [None, y_dims[1]], name="Y_class_labels")
    in_z = tf.placeholder(tf.int32, [None, z_dims[1]], name="Z_char_labels")

with tf.name_scope("Variables") as scope:
    W_class = tf.Variable(tf.random_normal(
        [hidden_size, y_dims[1]]), name="class_weight")
    b_class = tf.Variable(tf.random_normal([1, y_dims[1]]), name="class_bias")

    W_char = tf.Variable(tf.random_normal(
        [hidden_size, z_dims[1]]), name="char_weight")
    b_char = tf.Variable(tf.random_normal([1, z_dims[1]]), name="char_bias")

print("Creating the model and optimizers")
with tf.name_scope("optimizers"):
    class_logits = RNN(in_x, W_class, b_class, name="class_cell")
    class_pred = tf.nn.softmax(class_logits, name="class_prediction")
    class_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=class_logits, labels=in_y), name="class_cost")
    class_optimizer = tf.train.RMSPropOptimizer(0.01).minimize(class_cost)

    char_logits = RNN(in_x, W_char, b_char, name="char_cell")
    char_pred = tf.nn.softmax(char_logits, name="char_prediction")
    char_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=char_logits, labels=in_z), name="char_cost")
    char_optimizer = tf.train.RMSPropOptimizer(0.01).minimize(char_cost)

print("Setting up accuracy")
with tf.name_scope("accuracy"):
    max_class_pred = tf.argmax(class_pred, 1)
    class_correct_pred = tf.equal(max_class_pred, tf.argmax(in_y, 1))
    class_accuracy = tf.reduce_mean(
        tf.cast(class_correct_pred, tf.float32), name="class_accuracy")

    max_char_pred = tf.argmax(char_pred, 1)
    char_correct_pred = tf.equal(max_char_pred, tf.argmax(in_z, 1))
    char_accuracy = tf.reduce_mean(
        tf.cast(char_correct_pred, tf.float32), name="char_accuracy")

with tf.Session() as sess:
    print("Starting Session")
    writer = tf.summary.FileWriter('./text_gen', sess.graph)
    sess.run(tf.global_variables_initializer())
    print("Starting training")
    for i in range(iterations):
        bx, by, bz = gen.get_batch()

        feed_dict = {in_x: bx, in_y: by}
        _class_acc, _class_cost, _class_opt = sess.run(
            [class_accuracy, class_cost, class_optimizer], feed_dict=feed_dict)

        feed_dict = {in_x: bx, in_z: bz}
        _char_acc, _char_cost, _char_opt = sess.run(
            [char_accuracy, char_cost, char_optimizer], feed_dict=feed_dict)

        info = "class cost: {0:.4} char cost: {1:.4}".format(
            _class_cost, _char_cost)

        if i % batches == 0:
            print(info)

            show_len = 100
            start = 200
            seed_text = gen.raw_datatset[start:start + seq_length]
            gen_seq = []
            print(seed_text)
            seed_text = list(seed_text)
            for sh in range(show_len):

                seed_int = [gen.char2int[s] for s in seed_text]
                seedvec = np.array(seed_int, dtype=np.float32).reshape(
                    (1, seq_length, 1)) / 350

                _max_class_pred = sess.run(
                    [max_class_pred], feed_dict={in_x: seedvec})
                _max_char_pred = sess.run(
                    [max_char_pred], feed_dict={in_x: seedvec})
                cs = int(_max_class_pred[0][0]) * 10
                cr = int(_max_char_pred[0][0])
                char_int = cs + cr
                if char_int not in gen.int2char:
                    char_int = 350
                gen_seq.append(char_int)
                seed_text.append(gen.int2char[char_int])
                seed_text = seed_text[1:]

            gen_seq = ''.join([gen.int2char[g] for g in gen_seq])
            print(gen_seq)

    writer.close()
