#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np

FILE_PATH = '/home/swq/Downloads/flownet2-tf/data/FlyingChairs_release/data'
LIST_PATH = '/home/swq/Downloads/flownet2-tf/data'
LOG_PATH = '/home/swq/Documents/optic_flow_network/logs7'
CHECKPOINTS_PATH = '/home/swq/Documents/optic_flow_network/checkpoints7'

'''batch_size = 16就大了(GTX960)'''
BATCH_SIZE = 8
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3
FLO_CHANNELS = 2

INITIAL_LEARNING_RATE = 1e-4
MAX_STEPS = 100000
