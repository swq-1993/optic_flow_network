#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import util
import net_structure
from timer import Timer
import config
import datetime
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

batch_size = config.BATCH_SIZE
initial_learning_rate = config.INITIAL_LEARNING_RATE
max_step = config.MAX_STEPS
log_dir = config.LOG_PATH
checkpoints = config.CHECKPOINTS_PATH
momentum = 0.9
momentum2 = 0.999

val_loss_record = []
average_loss_record = []
input_data = {}


def train():
    trainset = util.get_train_file()
    img1_feed, img2_feed, flo_feed = trainset.build_batch(batch_size)
    learn_rate_feed = initial_learning_rate
    with tf.variable_scope("queue"):
        q = tf.FIFOQueue(capacity=5, dtypes=tf.float32)  # enqueue 5 batches
        enqueue_op = q.enqueue([img1_feed, img2_feed, flo_feed, learn_rate_feed])
        thread_number = 1
        queue_runner = tf.train.QueueRunner(q, [enqueue_op] * thread_number)
        tf.train.add_queue_runner(qr=queue_runner)
        input_data = q.dequeue()  # It replaces our input placeholder

    with tf.variable_scope("get_data"):

        # img1_placeholder, img2_placeholder, flo_placeholder = net_structure.placeholder_inputs()
        # learn_rate_placeholder = tf.placeholder(tf.float32, shape=())
        # predict = net_structure.net_structure(img1_placeholder, img2_placeholder)
        predict = net_structure.net_structure(input_data[0], input_data[1])
        # loss = net_structure.loss(flo_placeholder, predict)
        loss = net_structure.loss(input_data[2], predict)
        global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(input_data[3], momentum, momentum2)
    train_op = optimizer.minimize(loss, global_step=global_step)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    train_time = Timer()

    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    logfile = os.path.join(log_dir, 'train_average_Loss16.txt')

    loss_sum = 0

    # train_checkpoints = os.path.join(checkpoints, 'model.ckpt-19999')
    # saver.restore(sess, train_checkpoints)

    for step in xrange(max_step):
        train_time.tic()
        # feed_dict = net_structure.fill_feed_dict(trainset, img1_placeholder, img2_placeholder, flo_placeholder)
        # feed_dict[learn_rate_placeholder] = initial_learning_rate

        if step > 10000:
            learning_rate = (0.5 ** ((step - 10000) / 10000)) * initial_learning_rate
            input_data[3] = learning_rate

        _, loss_value = sess.run([train_op, loss])

        train_time.toc()

        loss_sum += loss_value

        if step % 100 == 0:
            log_info = ('{} Epoch: {}, step: {}, learning rate: {},'
                       'Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Remain: {}').format(
                datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                trainset.epochs_completed,
                int(step),
                input_data[3],
                loss_value,
                train_time.average_time,
                train_time.remain(step, max_step)
            )
            print log_info

        if (step + 1) % 100 == 0 or (step + 1) == max_step:
            f = open(logfile, 'a')
            print 'Average loss %0.04f' % (loss_sum / 100)
            f.write('loss_average:' + str(loss_sum / 100) + '\n')
            average_loss_record.append(loss_sum / 100)
            loss_sum = 0
            f.close()

        if (step + 1) % 500 == 0 or (step + 1) == max_step:
            checkpoint_file = os.path.join(checkpoints, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

        coord.request_stop()
        coord.join(threads)

    plt.figure(1)
    plt.plot(average_loss_record)
    plt.xlabel('iteration')
    plt.ylabel('train_loss')
    plt.title('train loss')
    plt.savefig('average_train_loss.png')


if __name__ == '__main__':
    train()
