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


out_path = config.OUT_PATH
test_path = config.TEST_PATH
checkpoints_path = config.CHECKPOINTS_PATH


def read_img(imgname):
    image = cv2.imread(imgname)
    image = image.astype(np.float32) / 255
    return image


def create_batch(name_list):
    array = []
    for name in name_list:
        array.append(read_img(name))
    array = np.asarray(array)
    return array


def run_test(img1, img2, test_out_path, test_checkpoints, save_image=True, save_flo=False):
    prediction = net_structure.net_structure(img1, img2)[3]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if test_checkpoints:
            saver.restore(sess, test_checkpoints)
            print 'restored checkpoints'

        pred_flow = sess.run(prediction)[0, :, :, :]
        unique_name = 'flow-' + str(uuid.uuid4())
        if save_image:
            flow_img = flow_to_image(pred_flow)
            full_out_path = os.path.join(test_out_path, unique_name + '.png')
            print unique_name
            imsave(full_out_path, flow_img)

        if save_flo:
            full_out_path = os.path.join(test_out_path, unique_name + '.flo')
            write_flow(pred_flow, full_out_path)


if __name__ == '__main__':
    img1 = '0img0.ppm'
    img2 = '0img1.ppm'
    img1_path = os.path.join(test_path, img1)
    img2_path = os.path.join(test_path, img2)

    batch_img1 = create_batch([img1_path])
    batch_img2 = create_batch([img2_path])

    # test_batch_img1 = test_data.create_batch(True, img1_path[0])
    # test_batch_img2 = test_data.create_batch(True, img2_path)

    checkpoints = os.path.join(checkpoints_path, 'model.ckpt-49999')
    run_test(batch_img1, batch_img2, out_path, checkpoints)
