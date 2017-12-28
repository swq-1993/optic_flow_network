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
    boundary_a_train_path = []
    boundary_b_train_path = []
    flow_train_path = []
    for i in train_idxs:
        image_a_train_path.append(os.path.join(file_path, '%05d_img1.ppm' % (i + 1)))
        image_b_train_path.append(os.path.join(file_path, '%05d_img2.ppm' % (i + 1)))
        boundary_a_train_path.append(os.path.join(file_path, '%05d_flow.png') % (i + 1))
        boundary_b_train_path.append(os.path.join(file_path, '%05d_flow2.png') % (i + 1))
        flow_train_path.append(os.path.join(file_path, '%05d_flow.flo' % (i + 1)))
    trainset = Data([image_a_train_path, image_b_train_path, boundary_a_train_path, boundary_b_train_path, flow_train_path])
    return trainset


def get_val_file():
    train_idxs, val_idxs = load_train_val_split()
    image_a_val_path = []
    image_b_val_path = []
    flow_val_path = []
    for i in val_idxs:
        image_a_val_path.append(os.path.join(file_path, '%05d_img1.ppm' % (i + 1)))
        image_b_val_path.append(os.path.join(file_path, '%05d_img2.ppm' % (i + 1)))
        flow_val_path.append(os.path.join(file_path, '%05d_flow.flo' % (i + 1)))
    valset = Data([image_a_val_path, image_b_val_path, flow_val_path])
    return valset


class Data(object):
    def __init__(self, nameset):
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = len(nameset[0])
        self.img1 = np.array(nameset[0])
        self.img2 = np.array(nameset[1])
        self.boundary1 = np.array(nameset[2])
        self.boundary2 = np.array(nameset[3])
        self.flo = np.array(nameset[4])

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

    def read_img(self, imgname):
        image = cv2.imread(imgname)
        image = image.astype(np.float32) / 255
        return image

    def create_batch(self, eval_img, name_list):
        array = []
        for name in name_list:
            if eval_img:
                array.append(self.read_img(name))
            else:
                array.append(self.read_flo(name))
        array = np.asarray(array)
        return array

    def get_batch(self, batch_size):
        # get data index depends on the length of batch_size
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            start = 0
            self.index_in_epoch = batch_size
            # start next epoch
            self.epochs_completed += 1
            # shuffle data
            perm = np.arange(self.num_examples)
            print perm
            np.random.shuffle(perm)
            print perm
            self.img1 = self.img1[perm]
            self.img2 = self.img2[perm]
            self.boundary1 = self.boundary1[perm]
            self.boundary2 = self.boundary2[perm]
            self.flo = self.flo[perm]
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        # get batch
        batch_img1 = self.create_batch(True, self.img1[start:end])
        batch_img2 = self.create_batch(True, self.img2[start:end])
        batch_boundary1 = self.create_batch(True, self.boundary1[start:end])
        batch_boundary2 = self.create_batch(True, self.boundary2[start:end])
        batch_flo = self.create_batch(False, self.flo[start:end])

        return batch_img1, batch_img2, batch_boundary1, batch_boundary2, batch_flo

    def build_batch(self, batch_size):
        img1 = self.create_batch(True, self.img1)
        img2 = self.create_batch(True, self.img2)
        flo = self.create_batch(False, self.flo)
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        img1_batch, img2_batch, flo_batch = tf.train.shuffle_batch([img1, img2, flo],
                                                                   batch_size=batch_size,
                                                                   capacity=capacity,
                                                                   min_after_dequeue=min_after_dequeue)
        return img1_batch, img2_batch, flo_batch


if __name__ == '__main__':
    trainset = get_train_file()
    batch_img1, batch_img2, batch_boundary1, batch_boundary2, batch_flo = trainset.get_batch(6)
    batch_img1, batch_img2, batch_boundary1, batch_boundary2, batch_flo = trainset.get_batch(6)
    for i in range(6):
        img = batch_img1[i]
        cv2.imshow('img'+str(i), img.astype(np.uint8))
        cv2.waitKey(0)
    pass
