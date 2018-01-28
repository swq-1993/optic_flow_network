#!/usr/bin/env python
# -*- coding: UTF-8 -*-

FILE_PATH = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
LIST_PATH = '/home/swq/Downloads/flownet2-tf/data'
LOG_PATH = '/home/swq/Documents/optic_flow_network/logs11'
CHECKPOINTS_PATH = '/home/swq/Documents/optic_flow_network/checkpoints11'

OUT_PATH = '/home/swq/Documents/optic_flow_network/out2'
TEST_PATH = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'

'''batch_size = 16就大了(GTX960)'''
BATCH_SIZE = 1
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3
FLO_CHANNELS = 2

INITIAL_LEARNING_RATE = 1e-4
MAX_STEPS = 100000
