# #!/usr/bin/env python
# # encoding: utf-8
#
# '''
#
# @author: HuangQiang
# @contact: huangqiang97@yahoo.com
# @project:
# @file: test_env.py
# @time: 2018/10/14 0:49
# @desc:
# @version:
#
# '''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import urllib
import urllib3
import scipy
import bs4

hello=tf.constant('tensorflow')
with tf.Session() as sess:
    print(sess.run(hello))


num_array=np.array([1,2,3,4,5,6,7,8],np.uint8)
print(num_array)
img_0=cv2.imread('F:/VisualCodeProject/learn_tensorflow_opencv/src/0.jpg',1)
cv2.imshow('F:/VisualCodeProject/learn_tensorflow_opencv/src/0.jpg',img_0)
x=np.linspace(0,7,8)
plt.plot(x,num_array,'r')
plt.show()
cv2.waitKey(0)


# print('fuck')
# import cv2
# import numpy as np
# import tensorflow as tf
#
# abc=np.zeros([2,3],np.uint8)
# print(abc)
# tf_var=tf.constant('1111',tf.uint16)
# with tf.Session() as sess:
#     print(sess.run(tf_var))
