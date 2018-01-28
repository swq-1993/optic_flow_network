#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import data
import net_structure
from timer import Timer
import config
import datetime
import os
import matplotlib.pyplot as plt
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


def run_val(sess, img1_palceholder, img2_placeholder, flo_placeholder, loss, validation):
    loss_count = 0
    step_per_epoch = validation.num_examples // batch_size
    num_examples = step_per_epoch * batch_size
    for step in xrange(step_per_epoch):
        feed_dict = net_structure.fill_feed_dict(validation, img1_palceholder, img2_placeholder, flo_placeholder)
        loss_count += sess.run(loss, feed_dict=feed_dict)
    average = float(loss_count) / num_examples
    val_loss_record.append(average)
    print 'Num examples: %d Average loss %0.04f' % (num_examples, average)


def train():
    trainset = data.get_train_file()
    valset = data.get_val_file()

    img1_placeholder, img2_placeholder, flo_placeholder = net_structure.placeholder_inputs()
    learn_rate_placeholder = tf.placeholder(tf.float32, shape=())
    deconvflow5, deconvflow4, deconvflow3, prediction = net_structure.net_structure(img1_placeholder, img2_placeholder)
    loss = net_structure.comput_loss(deconvflow5, deconvflow4, deconvflow3, prediction, flo_placeholder)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=2000, decay_rate=0.1)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate_placeholder, momentum, momentum2)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    train_time = Timer()

    sess.run(init)
    logfile = os.path.join(log_dir, 'train_average_Loss.txt')

    loss_sum = 0
    train_checkpoints = os.path.join(checkpoints, 'model.ckpt-11499')
    saver.restore(sess, train_checkpoints)

    for step in xrange(23600, max_step):
        train_time.tic()
        feed_dict = net_structure.fill_feed_dict(trainset, img1_placeholder, img2_placeholder, flo_placeholder)
        feed_dict[learn_rate_placeholder] = initial_learning_rate

        if step > 30000:
            learning_rate = (0.5 ** ((step - 30000) / 10000)) * initial_learning_rate
            feed_dict[learn_rate_placeholder] = learning_rate

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        train_time.toc()

        loss_sum += loss_value

        if step % 100 == 0:
            log_info = ('{} Epoch: {}, step: {}, learning rate: {},'
                       'Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Remain: {}').format(
                datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                trainset.epochs_completed,
                int(step),
                # learning_rate.eval(session=sess),
                feed_dict[learn_rate_placeholder],
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
        '''暂时不跑验证集了'''
        #     print 'Validation Data Eval:'
        #     run_val(sess, img1_placeholder, img2_placeholder, flo_placeholder, loss, valset)

    plt.figure(1)
    plt.plot(average_loss_record)
    plt.xlabel('iteration')
    plt.ylabel('train_loss')
    plt.title('train loss')
    plt.savefig('average_train_loss.png')

    # plt.figure(2)
    # plt.plot(val_loss_record)
    # plt.xlabel('iteration')
    # plt.ylabel('val_loss')
    # plt.title('val loss')
    # plt.savefig('val_loss.png')


if __name__ == '__main__':
    train()
