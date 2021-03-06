#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import net_structure
from timer import Timer
import config
import datetime
from scipy.misc import imsave
import cv2
import uuid
from flowlib import flow_to_image, write_flow
import os
import numpy as np
import dataset_configs
import data_load
import data


out_path = '/home/swq/Documents/optic_flow_network/out'
test_path = config.TEST_PATH
initial_learning_rate = config.INITIAL_LEARNING_RATE
checkpoints_path = '/home/swq/Documents/optic_flow_network/checkpoints9'
val_loss_record = []

list_path = config.LIST_PATH
TRAIN = 1
VAL = 2
momentum = 0.9
momentum2 = 0.999
loss_record_file = 'flownet_c_opti_loss.txt'


def load_train_val_split():
    train_val_split = os.path.join(list_path, 'FlyingChairs_train_val.txt')
    train_val_split = np.loadtxt(train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)
    return train_idxs, val_idxs


def read_img(imgname):
    image = cv2.imread(imgname)
    image = image.astype(np.float32)
    return image


def create_batch(name_list):
    array = []
    for name in name_list:
        array.append(read_img(name))
    array = np.asarray(array)
    return array


def run_test(img1, img2, test_out_path, test_checkpoints, save_image=False, save_flo=True):
    prediction = net_structure.net_structure(img1, img2)
    pred_flow = prediction['flow']
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if test_checkpoints:
            saver.restore(sess, test_checkpoints)
            print 'restored checkpoints'

        pred_flow = sess.run(pred_flow)[0, :, :, :]
        unique_name = 'flow_predict_old' + str(uuid.uuid4())
        if save_image:
            flow_img = flow_to_image(pred_flow)
            full_out_path = os.path.join(test_out_path, unique_name + '.png')
            print unique_name
            imsave(full_out_path, flow_img)

        if save_flo:
            full_out_path = os.path.join(test_out_path, unique_name + '.flo')
            write_flow(pred_flow, full_out_path)


def test(checkpoint, input_a_path, input_b_path, img_num, out_path, save_image=True, save_flo=True):
    input_a = cv2.imread(input_a_path)
    input_b = cv2.imread(input_b_path)
    # input_a = cv2.resize(input_a, (512, 384))
    # input_b = cv2.resize(input_b, (512, 384))
    # Convert from RGB -> BGR
    input_a = input_a[..., [2, 1, 0]]
    input_b = input_b[..., [2, 1, 0]]

    # Scale from [0, 255] -> [0.0, 1.0] if needed
    # if input_a.max() > 1.0:
    #     input_a = input_a / 255.0
    # if input_b.max() > 1.0:
    #     input_b = input_b / 255.0
    input_a = input_a / 255.0
    input_b = input_b / 255.0
    # TODO: This is a hack, we should get rid of this
    # training_schedule = LONG_SCHEDULE

    inputs = {
        'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
        'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
    }

    batch_img1 = create_batch([img1_path])
    batch_img2 = create_batch([img2_path])
    predictions = net_structure.net_structure(batch_img1, batch_img2)
    pred_flow = predictions['flow']

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    saver.restore(sess, checkpoint)
    pred_flow = sess.run(pred_flow)[0, :, :, :]

    # unique_name = 'flow-' + str(uuid.uuid4())
    unique_name = 'flow_predict_old_' + img_num
    print unique_name
    if save_image:
        flow_img = flow_to_image(pred_flow)
        full_out_path = os.path.join(out_path, unique_name + '.png')
        imsave(full_out_path, flow_img)

    if save_flo:
        full_out_path = os.path.join(out_path, unique_name + '.flo')
        write_flow(pred_flow, full_out_path)


if __name__ == '__main__':
    index = 300
    img_num = '%05d' % index
    img1 = img_num + '_img1.ppm'
    img2 = img_num + '_img2.ppm'
    img1_path = os.path.join(test_path, img1)
    img2_path = os.path.join(test_path, img2)

    batch_img1 = create_batch([img1_path])
    batch_img2 = create_batch([img2_path])

    # test_batch_img1 = test_data.create_batch(True, img1_path[0])
    # test_batch_img2 = test_data.create_batch(True, img2_path)

    checkpoints = os.path.join(checkpoints_path, 'model.ckpt-99999')
    # run_test(batch_img1, batch_img2, out_path, checkpoints)
    test(checkpoints, img1_path, img2_path, img_num, out_path)
