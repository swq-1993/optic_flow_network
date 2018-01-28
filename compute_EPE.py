import tensorflow as tf
from data_load_boundary import load_batch
import dataset_configs
import os
import numpy as np
import cv2
import struct
import config

ground_truth_path = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
old_predict_path = './out'
new_predict_path = './out2'
canny_predict_path = './canny_out'
list_path = config.LIST_PATH
TRAIN = 1
VAL = 2
contract_loss_file = 'contract_loss_val.txt'


def load_train_val_split():
    train_val_split = os.path.join(list_path, 'FlyingChairs_train_val.txt')
    train_val_split = np.loadtxt(train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)
    return train_idxs, val_idxs


def read_flo(floname):
    f = open(floname, "rb")
    data = f.read()
    f.close()
    width = struct.unpack('@i', data[4:8])[0]
    height = struct.unpack('@i', data[8:12])[0]
    flodata = np.zeros((height, width, 2))
    for i in range(width * height):
        data_u = struct.unpack('@f', data[12 + 8 * i:16 + 8 * i])[0]
        data_v = struct.unpack('@f', data[16 + 8 * i:20 + 8 * i])[0]
        n = int(i / width)
        k = np.mod(i, width)
        flodata[n, k, :] = [data_u, data_v]
    return flodata


def read_img(imgname):
    image = cv2.imread(imgname)
    image = image.astype(np.float32)
    return image


def create_batch(eval_img, name_list):
    array = []
    for name in name_list:
        if eval_img:
            array.append(read_img(name))
        else:
            array.append(read_flo(name))
    array = np.asarray(array)
    return array


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    sample_height = predictions.shape.as_list()[2]
    sample_width = predictions.shape.as_list()[3]

    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)

        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 4)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples / (sample_height * sample_width)


if __name__ == '__main__':
    old_loss_sum = 0.0
    new_loss_sum = 0.0
    canny_loss_sum = 0.0
    for i in xrange(150, 300):
        name_num = '%05d' % (i + 1)
        ground_truth_name = name_num + '_flow.flo'
        old_predict_name = 'flow_predict_old_' + name_num + '.flo'
        new_predict_name = 'flow_predict_new_' + name_num + '.flo'
        canny_predict_name = 'flow_predict_canny_' + name_num + '.flo'

        ground_truth = os.path.join(ground_truth_path, ground_truth_name)
        old_predict = os.path.join(old_predict_path, old_predict_name)
        new_predict = os.path.join(new_predict_path, new_predict_name)
        canny_predict = os.path.join(canny_predict_path, canny_predict_name)

        batch_ground_truth = create_batch(False, [ground_truth])
        batch_old_predict = create_batch(False, [old_predict])
        batch_new_predict = create_batch(False, [new_predict])
        batch_canny_predict = create_batch(False, [canny_predict])

        batch_ground_truth = tf.expand_dims(tf.constant(batch_ground_truth, dtype=tf.float32), 0)
        batch_old_predict = tf.expand_dims(tf.constant(batch_old_predict, dtype=tf.float32), 0)
        batch_new_predict = tf.expand_dims(tf.constant(batch_new_predict, dtype=tf.float32), 0)
        batch_canny_predict = tf.expand_dims(tf.constant(batch_canny_predict, dtype=tf.float32), 0)

        old_loss = average_endpoint_error(batch_ground_truth, batch_old_predict)
        new_loss = average_endpoint_error(batch_ground_truth, batch_new_predict)
        canny_loss = average_endpoint_error(batch_ground_truth, batch_canny_predict)
        with tf.Session() as sess:
            old_loss, new_loss, canny_loss = sess.run([old_loss, new_loss, canny_loss])
            # old_loss = sess.run(old_loss)
        print old_loss
        print new_loss
        print canny_loss
        old_loss_sum += old_loss
        new_loss_sum += new_loss
        canny_loss_sum += canny_loss
        # print old_loss - new_loss
        # print old_loss - canny_loss
        # f = open(contract_loss_file, 'a')
        # f.write(name_num + ' old_loss:  ' + str(old_loss))
        # f.write(' new_loss:  ' + str(new_loss))
        # f.write(' canny_loss:  ' + str(canny_loss))
        # f.write(' motion_diff:  ' + str(old_loss - new_loss))
        # f.write(' canny_diff:  ' + str(old_loss - canny_loss) + '\n')
        # f.close()
    f = open(contract_loss_file, 'a')
    f.write('total_old_loss_sum:' + str(old_loss_sum) + '\n')
    f.write('total_new_loss_sum:' + str(new_loss_sum) + '\n')
    f.write('total_canny_loss_sum: ' + str(canny_loss_sum) + '\n')
    f.close()
    # print val_idxs
