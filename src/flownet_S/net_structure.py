#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import config
import data

batch_size = config.BATCH_SIZE
img_height = config.IMAGE_HEIGHT
img_width = config.IMAGE_WIDTH
img_channels = config.IMAGE_CHANNELS
flo_channels = config.FLO_CHANNELS
loss_weight_schedule = config.loss_weights_schedule


def placeholder_inputs():
    img1_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))
    img2_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))
    flo_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, flo_channels))
    return img1_placeholder, img2_placeholder, flo_placeholder


def fill_feed_dict(data, img1_pl, img2_pl, flo_pl):
    img1_feed, img2_feed, flo_feed = data.get_batch(batch_size)
    # test
    # cv2.imshow('img1', img1_feed[3].astype(np.uint8))
    # cv2.imshow('img2', img2_feed[1].astype(np.uint8))
    # cv2.waitKey()

    feed_dict = {img1_pl: img1_feed,
                 img2_pl: img2_feed,
                 flo_pl: flo_feed
                 }
    return feed_dict


def net_structure(img1, img2):
    concat1 = tf.concat([img1, img2], axis=3, name='concat1')
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu):
        conv1 = slim.conv2d(concat1, 64, [7, 7], stride=2)
        conv2 = slim.conv2d(conv1, 128, [5, 5], stride=2)
        conv3 = slim.conv2d(conv2, 256, [5, 5], stride=2)
        conv3_1 = slim.conv2d(conv3, 256, [3, 3], stride=1)
        conv4 = slim.conv2d(conv3_1, 512, [3, 3], stride=2)
        conv4_1 = slim.conv2d(conv4, 512, [3, 3], stride=1)
        conv5 = slim.conv2d(conv4_1, 512, [3, 3], stride=2)
        conv5_1 = slim.conv2d(conv5, 512, [3, 3], stride=1)
        conv6 = slim.conv2d(conv5_1, 1024, [3, 3], stride=2)

    '''start refinement'''
    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu):

        deconv5 = slim.conv2d_transpose(conv6, 512, [1, 1], stride=2)
        concat5 = tf.concat([conv5_1, deconv5], axis=3, name='concat5')

        flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1)
        deconv4 = slim.conv2d_transpose(concat5, 256, [3, 3], stride=2)
        deconvflow5 = slim.conv2d_transpose(flow5, 2, [2, 2], stride=2)
        concat4 = tf.concat([conv4_1, deconv4, deconvflow5], axis=3, name='concat4')

        flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1)
        deconv3 = slim.conv2d_transpose(concat4, 128, [3, 3], stride=2)
        deconvflow4 = slim.conv2d_transpose(flow4, 2, [2, 2], stride=2)
        concat3 = tf.concat([conv3_1, deconv3, deconvflow4], axis=3, name='concat3')

        flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1)
        deconv2 = slim.conv2d_transpose(concat3, 64, [3, 3], stride=2)
        deconvflow3 = slim.conv2d_transpose(flow3, 2, [2, 2], stride=2)
        concat2 = tf.concat([conv2, deconv2, deconvflow3], axis=3, name='concat2')

        prediction = slim.conv2d(concat2, 2, [3, 3], stride=1)
        # prediction = slim.conv2d_transpose(prediction, 2, [2, 2], stride=2)
    return deconvflow5, deconvflow4, deconvflow3, prediction


def tensor_loss(tensor, name):
    loss = tf.reduce_mean(tf.abs(tensor), name=name)
    tf.add_to_collection('losses', loss)
    tf.summary.scalar('losses'+name, loss)
    return loss


def comput_loss(deconvflow5, deconvflow4, deconvflow3, prediction, flo):
    flo5 = tf.image.resize_images(flo, [24, 32])
    loss5 = 0.2 * tensor_loss(flo5 - deconvflow5, 'loss5')
    flo4 = tf.image.resize_images(flo, [48, 64])
    loss4 = 0.2 * tensor_loss(flo4 - deconvflow4, 'loss4')
    flo3 = tf.image.resize_images(flo, [96, 128])
    loss3 = 0.2 * tensor_loss(flo3 - deconvflow3, 'loss3')
    flo4 = tf.image.resize_images(flo, [192, 256])
    loss2 = 1 * tensor_loss(flo4 - prediction, 'loss2')

    total_loss = tf.add_n([loss5, loss4, loss3, loss2], name='total_loss')
    tf.summary.scalar('losses'+'total_loss', total_loss)
    return total_loss


if __name__ == '__main__':
    trainset = data.get_train_file()
    batch_img1, batch_img2, batch_flo = trainset.get_batch(4)
    net_structure(batch_img1, batch_img2)

# if __name__ == '__main__':
#     img1_placeholder, img2_placeholder, flo_placeholder = placeholder_inputs()
#     trainset = get_train_file()
#     feed_dict = feed_dict(trainset, img1_placeholder, img2_placeholder, flo_placeholder)
