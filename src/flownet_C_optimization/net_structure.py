#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import config
import cv2
import numpy as np
from util import *
import correlation
import downsample

batch_size = config.BATCH_SIZE
img_height = config.IMAGE_HEIGHT
img_width = config.IMAGE_WIDTH
img_channels = config.IMAGE_CHANNELS
flo_channels = config.FLO_CHANNELS

weight_decay = 0.0004


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples


def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


def placeholder_inputs():
    img1_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))
    img2_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))
    flo_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, flo_channels))
    return img1_placeholder, img2_placeholder, flo_placeholder


def fill_feed_dict(data, img1_pl, img2_pl, flo_pl):
    img1_feed, img2_feed, flo_feed = data.build_batch(batch_size)
    # test
    # cv2.imshow('img1', img1_feed[3].astype(np.uint8))
    # cv2.imshow('img2', img2_feed[1].astype(np.uint8))
    # cv2.waitKey()
    feed_dict = {img1_pl: img1_feed, img2_pl: img2_feed, flo_pl: flo_feed}
    return feed_dict


def net_structure(img1, img2):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        # He (aka MSRA) weight initialization
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=LeakyReLU,
                        # We will do our own padding to match the original Caffe code
                        padding='VALID'):
        weights_regularizer = slim.l2_regularizer(weight_decay)
        with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
            with slim.arg_scope([slim.conv2d], stride=2):
                conv_a_1 = slim.conv2d(pad(img1, 3), 64, 7, scope='conv1')
                conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope='conv2')
                conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope='conv3')

                conv_b_1 = slim.conv2d(pad(img2, 3),
                                       64, 7, scope='conv1', reuse=True)
                conv_b_2 = slim.conv2d(pad(conv_b_1, 2), 128, 5, scope='conv2', reuse=True)
                conv_b_3 = slim.conv2d(pad(conv_b_2, 2), 256, 5, scope='conv3', reuse=True)

                # Compute cross correlation with leaky relu activation
                cc = correlation.correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
                cc_relu = LeakyReLU(cc)

            # Combine cross correlation results with convolution of feature map A
            netA_conv = slim.conv2d(conv_a_3, 32, 1, scope='conv_redir')
            # Concatenate along the channels axis
            net = tf.concat([netA_conv, cc_relu], axis=3)

            conv3_1 = slim.conv2d(pad(net), 256, 3, scope='conv3_1')
            with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
            conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
            conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

            """ START: Refinement Network """
            with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
                                            scope='predict_flow6',
                                            activation_fn=None)
                deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4,
                                                        stride=2,
                                                        scope='deconv5'))
                upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
                                                                  stride=2,
                                                                  scope='upsample_flow6to5',
                                                                  activation_fn=None))
                concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
                                            scope='predict_flow5',
                                            activation_fn=None)
                deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,
                                                        stride=2,
                                                        scope='deconv4'))
                upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
                                                                  stride=2,
                                                                  scope='upsample_flow5to4',
                                                                  activation_fn=None))
                concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
                                            scope='predict_flow4',
                                            activation_fn=None)
                deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,
                                                        stride=2,
                                                        scope='deconv3'))
                upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                  stride=2,
                                                                  scope='upsample_flow4to3',
                                                                  activation_fn=None))
                concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
                                            scope='predict_flow3',
                                            activation_fn=None)
                deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,
                                                        stride=2,
                                                        scope='deconv2'))
                upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                  stride=2,
                                                                  scope='upsample_flow3to2',
                                                                  activation_fn=None))
                concat2 = tf.concat([conv_a_2, deconv2, upsample_flow3to2], axis=3)

                predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
                                            scope='predict_flow2',
                                            activation_fn=None)
            """ END: Refinement Network """

            flow = predict_flow2 * 20.0
            # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
            flow = tf.image.resize_bilinear(flow,
                                            tf.stack([img_height, img_width]),
                                            align_corners=True)

            return {
                'predict_flow6': predict_flow6,
                'predict_flow5': predict_flow5,
                'predict_flow4': predict_flow4,
                'predict_flow3': predict_flow3,
                'predict_flow2': predict_flow2,
                'flow': flow,
            }


def loss(flow, predictions):
    flow = flow * 0.05

    losses = []
    INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

    # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
    predict_flow6 = predictions['predict_flow6']
    size = [predict_flow6.shape[1], predict_flow6.shape[2]]
    downsampled_flow6 = downsample.downsample(flow, size)
    losses.append(average_endpoint_error(downsampled_flow6, predict_flow6))

    # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
    predict_flow5 = predictions['predict_flow5']
    size = [predict_flow5.shape[1], predict_flow5.shape[2]]
    downsampled_flow5 = downsample.downsample(flow, size)
    losses.append(average_endpoint_error(downsampled_flow5, predict_flow5))

    # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
    predict_flow4 = predictions['predict_flow4']
    size = [predict_flow4.shape[1], predict_flow4.shape[2]]
    downsampled_flow4 = downsample.downsample(flow, size)
    losses.append(average_endpoint_error(downsampled_flow4, predict_flow4))

    # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
    predict_flow3 = predictions['predict_flow3']
    size = [predict_flow3.shape[1], predict_flow3.shape[2]]
    downsampled_flow3 = downsample.downsample(flow, size)
    losses.append(average_endpoint_error(downsampled_flow3, predict_flow3))

    # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
    predict_flow2 = predictions['predict_flow2']
    size = [predict_flow2.shape[1], predict_flow2.shape[2]]
    downsampled_flow2 = downsample.downsample(flow, size)
    losses.append(average_endpoint_error(downsampled_flow2, predict_flow2))

    loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])

    # Return the 'total' loss: loss fns + regularization terms defined in the model
    return tf.losses.get_total_loss()


if __name__ == '__main__':
    trainset = get_train_file()
    batch_img1, batch_img2, batch_flo = trainset.get_batch(4)
    net_structure(batch_img1, batch_img2)
