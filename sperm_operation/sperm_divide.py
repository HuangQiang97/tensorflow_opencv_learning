#!/usr/bin/env python
# encoding: utf-8

'''

@author: HuangQiang
@contact: huangqiang97@yahoo.com
@project:
@file: sperm_divide.py
@time: 2018/10/18 10:21
@desc:
@version:

'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
img_src=cv2.imread('../img_lib/pic_5.jpg',0)
cv2.imshow('img_src',img_src)
cv2.waitKey(0)