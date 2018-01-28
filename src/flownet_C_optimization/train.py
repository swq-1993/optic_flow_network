#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import net_structure
from timer import Timer
import config
import datetime
import os
import matplotlib.pyplot as plt
import dataset_configs
import data_load
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


def train():
    # trainset = util.get_train_file()

    learn_rate_placeholder = tf.placeholder(tf.float32, shape=())
    input_a, input_b, flow = data_load.load_batch(dataset_configs.FLYING_CHAIRS_DATASET_CONFIG, 'train')

    predict = net_structure.net_structure(input_a, input_b)
    loss = net_structure.loss(flow, predict)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate_placeholder, momentum, momentum2)
    train_op = optimizer.minimize(loss, global_step=global_step)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_time = Timer()

        sess.run(init)
        logfile = os.path.join(log_dir, 'train_average_Loss16.txt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print "queue runners started"
        loss_sum = 0
        feed_dict = {}
        # train_checkpoints = os.path.join(checkpoints, 'model.ckpt-19999')
        # saver.restore(sess, train_checkpoints)

        for step in xrange(max_step):
            train_time.tic()
            feed_dict[learn_rate_placeholder] = initial_learning_rate

            if step > 10000:
                learning_rate = (0.5 ** ((step - 10000) / 10000)) * initial_learning_rate
                feed_dict[learn_rate_placeholder] = learning_rate

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # for i in range(6):
            #     img = input_b[i, :, :, :] * 255.0
            #     cv2.imshow('img' + str(i), img.astype(np.uint8))
            #     cv2.waitKey(0)

            train_time.toc()
            print train_time.average_time
            loss_sum += loss_value

            if step % 100 == 0:
                log_info = ('{} Epoch: {}, step: {}, learning rate: {},'
                           'Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Remain: {}').format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    # trainset.epochs_completed,
                    int(step),
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
