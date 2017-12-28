#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import config
import os
import struct
import tensorflow as tf

file_path = config.FILE_PATH
list_path = config.LIST_PATH

TRAIN = 1
VAL = 2

Height = config.IMAGE_HEIGHT
Width = config.IMAGE_WIDTH
channels = config.IMAGE_CHANNELS


def load_train_val_split():
    train_val_split = os.path.join(list_path, 'FlyingChairs_train_val.txt')
    train_val_split = np.loadtxt(train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)
    return train_idxs, val_idxs


def get_train_file():
    train_idxs, val_idxs = load_train_val_split()
    image_a_train_path = []
    image_b_train_path = []
    flow_train_path = []
    for i in train_idxs:
        image_a_train_path.append(os.path.join(file_path, '%05d_img1.ppm' % (i + 1)))
        image_b_train_path.append(os.path.join(file_path, '%05d_img2.ppm' % (i + 1)))
        flow_train_path.append(os.path.join(file_path, '%05d_flow.flo' % (i + 1)))
    trainset = Data([image_a_train_path, image_b_train_path, flow_train_path])
    return trainset

# def get_train_boundary_file():
#     train_idxs, val_idxs = load_train_val_split()
#     image_a_train_path = []
#     image_b_train_path = []
#     boundary_a_train_path = []
#     boundary_b_train_path = []
#     flow_train_path = []
#     for i in train_idxs:
#         image_a_train_path.append(os.path.join(file_path, '%05d_img1.ppm' % (i + 1)))
#         image_b_train_path.append(os.path.join(file_path, '%05d_img2.ppm' % (i + 1)))
#
#         flow_train_path.append(os.path.join(file_path, '%05d_flow.flo' % (i + 1)))
#     trainset = Data([image_a_train_path, image_b_train_path, flow_train_path])
#     return trainset


class Data(object):
    def __init__(self, nameset):
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = len(nameset[0])
        self.img1 = np.array(nameset[0])
        self.img2 = np.array(nameset[1])
        self.flo = np.array(nameset[2])

    def get_file_name(self):
        file_names = [(self.img1[i], self.img2[i], self.flo[i]) for i in range(0, self.num_examples)]
        return file_names

    def read_file(self, file_names, epochs):
        file_name_queue = tf.train.input_producer(file_names, element_shape=[3], num_epochs=epochs, shuffle=True)
        file_name = file_name_queue.dequeue()
        # print file_name
        img1_name, img2_name, flo_name = file_name[0], file_name[1], file_name[2]
        # print img1_name
        # img1_b = tf.read_file(img1_name)
        # img2_b = tf.read_file(img2_name)
        # img1 = tf.image.decode_image(img1_b)
        # img2 = tf.image.decode_image(img2_b)
        #
        # flo_b = tf.read_file(flo_name)
        # flo = tf.image.decode_image(flo_b)
        # return img1, img2, flo
        img1 = self.create_batch(True, img1_name)
        img2 = self.create_batch(True, img2_name)
        flo = self.create_batch(False, flo_name)
        return img1, img2, flo

    def read_img(self, imgname):
        image = cv2.imread(imgname)
        image = image.astype(np.float32) / 255
        return image

    def read_flo(self, floname):
        f = open(floname, "rb")
        data = f.read()
        f.close()
        width = struct.unpack('@i', data[4:8])[0]
        height = struct.unpack('@i', data[8:12])[0]
        flodata = np.zeros((height, width, 2))
        for i in range(width*height):
            data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
            data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
            n = int(i / width)
            k = np.mod(i, width)
            flodata[n, k, :] = [data_u, data_v]
        return flodata

    def create_batch(self, eval_img, name_list):
        array = []
        for name in name_list:
            if eval_img:
                array.append(self.read_img(name))
            else:
                array.append(self.read_flo(name))
        array = np.asarray(array)
        return array

    def image_preprocess(self, img1, img2, flo):
        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)

        flo = tf.cast(flo, tf.float32)

        orig_width = tf.shape(img1)[1]  # HWC,  [1]便指的是W
        img1_rsz = tf.image.resize_bilinear(img1[np.newaxis, :, :, :], [Height, Width])[0]
        img2_rsz = tf.image.resize_bilinear(img2[np.newaxis, :, :, :], [Height, Width])[0]
        flo_rsz = tf.image.resize_nearest_neighbor(flo[np.newaxis, :, :, :], [Height, Width])[0]
        flo_rsz = flo_rsz * Width / tf.to_float(orig_width)

        img1_rsz.set_shape([Height, Width, channels])
        img2_rsz.set_shape([Height, Width, channels])
        flo_rsz.set_shape([Height, Width, 2])

        return img1_rsz, img2_rsz, flo_rsz

    def build_batch(self, batch_size):
        file_name = self.get_file_name()
        img1, img2, flo = self.read_file(file_name, epochs=None)
        img1_rsz, img2_rsz, flo_rsz = self.image_preprocess(img1, img2, flo)
        min_after_dequeue = 40
        capacity = min_after_dequeue + 3 * batch_size
        # capacity = 50000
        img1_batch, img2_batch, flo_batch = tf.train.shuffle_batch([img1_rsz, img2_rsz, flo_rsz],
                                                                   batch_size=batch_size,
                                                                   capacity=capacity,
                                                                   num_threads=16,
                                                                   min_after_dequeue=min_after_dequeue)
        return img1_batch, img2_batch, flo_batch


if __name__ == '__main__':
    trainset = get_train_file()
    batch_img1, batch_img2, batch_flo = trainset.build_batch(6)
    for i in range(6):
        img = batch_img1[i]
        cv2.imshow('img'+str(i), img.astype(np.uint8))
        cv2.waitKey(0)
    pass
