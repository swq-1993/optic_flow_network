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
import data
import data_load_boundary
import dataset_configs


# out_path = config.OUT_PATH
out_path = '/home/swq/Documents/optic_flow_network/canny_out'
test_path = config.TEST_PATH
file_path = config.FILE_PATH
checkpoints_path = '/home/swq/Documents/optic_flow_network/checkpoints11'  # motion boundary
initial_learning_rate = config.INITIAL_LEARNING_RATE
list_path = config.LIST_PATH
boundary_path = '/home/swq/Documents/my_deeplab_resnet/flow_contours_output'
canny_contours_path = '/home/swq/Documents/my_deeplab_resnet/canny_contours'
loss_record_file = 'new_netstruct_loss.txt'


TRAIN = 1
VAL = 2
momentum = 0.9
momentum2 = 0.999


def load_train_val_split():
    train_val_split = os.path.join(list_path, 'FlyingChairs_train_val.txt')
    train_val_split = np.loadtxt(train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)
    return train_idxs, val_idxs


def read_img( imgname):
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


def test(checkpoint, input_a_path, input_b_path, boundary_a_path, boundary_b_path, img_num, out_path, save_image=True, save_flo=True):
    input_a = cv2.imread(input_a_path)
    input_b = cv2.imread(input_b_path)
    boundary_a = cv2.imread(boundary_a_path)
    boundary_b = cv2.imread(boundary_b_path)

    # input_a = cv2.resize(input_a, (512, 384))
    # input_b = cv2.resize(input_b, (512, 384))
    # Convert from RGB -> BGR
    input_a = input_a[..., [2, 1, 0]]
    input_b = input_b[..., [2, 1, 0]]
    boundary_a = boundary_a[..., [2, 1, 0]]
    boundary_b = boundary_b[..., [2, 1, 0]]

    # Scale from [0, 255] -> [0.0, 1.0] if needed
    # if input_a.max() > 1.0:
    #     input_a = input_a / 255.0
    # if input_b.max() > 1.0:
    #     input_b = input_b / 255.0
    input_a = input_a / 255.0
    input_b = input_b / 255.0
    boundary_a = boundary_a / 255.0
    boundary_b = boundary_b / 255.0

    # TODO: This is a hack, we should get rid of this
    # training_schedule = LONG_SCHEDULE

    inputs = {
        'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
        'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
    }

    batch_img1 = create_batch([img1_path])
    batch_img2 = create_batch([img2_path])
    batch_boundary_a = create_batch([boundary_a_path])
    batch_boundary_b = create_batch([boundary_b_path])
    predictions = net_structure.net_structure(batch_img1, batch_img2, batch_boundary_a, batch_boundary_b)
    pred_flow = predictions['flow']

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        pred_flow = sess.run(pred_flow)[0, :, :, :]

        unique_name = 'flow_predict_canny_' + img_num
        print unique_name
        if save_image:
            flow_img = flow_to_image(pred_flow)
            full_out_path = os.path.join(out_path, unique_name + '.png')
            imsave(full_out_path, flow_img)

        if save_flo:
            full_out_path = os.path.join(out_path, unique_name + '.flo')
            write_flow(pred_flow, full_out_path)


if __name__ == '__main__':
    index = 171
    img_num = '%05d' % index
    img1 = img_num + '_img1.ppm'
    img2 = img_num + '_img2.ppm'
    boundary_a = img_num + '_flow.png'
    boundary_b = img_num + '_flow2.png'
    img1_path = os.path.join(file_path, img1)
    img2_path = os.path.join(file_path, img2)
    boundary_a_path = os.path.join(canny_contours_path, boundary_a)
    boundary_b_path = os.path.join(canny_contours_path, boundary_b)

    batch_img1 = create_batch([img1_path])
    batch_img2 = create_batch([img2_path])
    batch_boundary_a = create_batch([boundary_a_path])
    batch_boundary_b = create_batch([boundary_b_path])

    checkpoints = os.path.join(checkpoints_path, 'model.ckpt-99999')
    # run_test(batch_img1, batch_img2, out_path, checkpoints)
    test(checkpoints, img1_path, img2_path, boundary_a_path, boundary_b_path, img_num, out_path)
