#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import copy
slim = tf.contrib.slim
import config
import dataset_configs

file_path = config.FILE_PATH
list_path = config.LIST_PATH

TRAIN = 1
VAL = 2

Height = config.IMAGE_HEIGHT
Width = config.IMAGE_WIDTH
channels = config.IMAGE_CHANNELS


class Image(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 image_key=None,
                 format_key=None,
                 shape=None,
                 channels=3,
                 dtype=tf.uint8,
                 repeated=False):
        """Initializes the image.
        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
          dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
              See tf.image.decode_image,
                  tf.decode_raw,
          repeated: if False, decodes a single image. If True, decodes a
            variable number of image strings from a 1D tensor of strings.
        """
        if not image_key:
            image_key = 'image/encoded'

        super(Image, self).__init__([image_key])
        self._image_key = image_key
        self._shape = shape
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]

        if self._repeated:
            return tf.functional_ops.map_fn(lambda x: self._decode(x),
                                            image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer)

    def _decode(self, image_buffer):
        """Decodes the image buffer.
        Args:
          image_buffer: The tensor representing the encoded image tensor.
        Returns:
          A tensor that represents decoded image of self._shape, or
          (?, ?, self._channels) if self._shape is not specified.
        """
        def decode_raw():
            """Decodes a raw image."""
            return tf.decode_raw(image_buffer, out_type=self._dtype)

        image = decode_raw()
        # image.set_shape([None, None, self._channels])
        if self._shape is not None:
            image = tf.reshape(image, self._shape)

        return image


def __get_dataset(dataset_config, split_name):
    """
    dataset_config: A dataset_config defined in datasets.py
    split_name: 'train'/'validate'
    """
    with tf.name_scope('__get_dataset'):
        if split_name not in dataset_config['SIZES']:
            raise ValueError('split name %s not recognized' % split_name)

        IMAGE_HEIGHT, IMAGE_WIDTH = dataset_config['IMAGE_HEIGHT'], dataset_config['IMAGE_WIDTH']
        reader = tf.TFRecordReader
        keys_to_features = {
            'image_a': tf.FixedLenFeature((), tf.string),
            'image_b': tf.FixedLenFeature((), tf.string),
            'boundary_a': tf.FixedLenFeature((), tf.string),
            'boundary_b': tf.FixedLenFeature((), tf.string),
            'flow': tf.FixedLenFeature((), tf.string),
        }
        items_to_handlers = {
            'image_a': Image(
                image_key='image_a',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'image_b': Image(
                image_key='image_b',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3),
            'boundary_a': Image(
                image_key='boundary_a',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3
            ),
            'boundary_b': Image(
                image_key='boundary_b',
                dtype=tf.float64,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                channels=3
            ),
            'flow': Image(
                image_key='flow',
                dtype=tf.float32,
                shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 2],
                channels=2),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        return slim.dataset.Dataset(
            data_sources=dataset_config['PATHS'][split_name],
            reader=reader,
            decoder=decoder,
            num_samples=dataset_config['SIZES'][split_name],
            items_to_descriptions=dataset_config['ITEMS_TO_DESCRIPTIONS'])


def load_batch(dataset_config, split_name):
    num_threads = 2
    reader_kwargs = {'options': tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)}

    with tf.name_scope('load_batch'):
        dataset = __get_dataset(dataset_config, split_name)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_threads,
            common_queue_capacity=100,
            common_queue_min=50,
            reader_kwargs=reader_kwargs)
        image_a, image_b, boundary_a, boundary_b, flow = data_provider.get(['image_a', 'image_b', 'boundary_a', 'boundary_b', 'flow'])
        image_a, image_b, boundary_a, boundary_b, flow = map(tf.to_float, [image_a, image_b, boundary_a, boundary_b, flow])

        image_as, image_bs, flows = map(lambda x: tf.expand_dims(x, 0), [image_a, image_b, flow])

        return tf.train.shuffle_batch([image_a, image_b, boundary_a, boundary_b, flow],
                                      # enqueue_many=True,
                                      batch_size=config.BATCH_SIZE,
                                      capacity=config.BATCH_SIZE * 16,
                                      min_after_dequeue=config.BATCH_SIZE * 4,
                                      num_threads=num_threads,
                                      allow_smaller_final_batch=False)


if __name__ == '__main__':
    image_a, image_b, boundary_a, boundary_b, flow = load_batch(dataset_configs.FLYING_CHAIRS_DATASET_CONFIG, 'train')
    pass
