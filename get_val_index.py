#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
import config
list_path = config.LIST_PATH
TRAIN = 1
VAL = 2
train_val_split = os.path.join(list_path, 'FlyingChairs_train_val.txt')
train_val_split = np.loadtxt(train_val_split)
# train_idxs = np.flatnonzero(train_val_split == TRAIN)
val_idxs = np.flatnonzero(train_val_split == VAL)
f = open("val_index.txt", "wb")
for i in val_idxs:
    f.write(str(i) + '\n')
f.close()

print val_idxs
