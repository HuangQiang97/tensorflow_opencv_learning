#!/usr/bin/env python
# encoding: utf-8

'''

@author: HuangQiang
@contact: huangqiang97@yahoo.com
@project:
@file: basic_operation.py
@time: 2018/10/14 11:40
@desc:
@version:

'''
import math
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
#导入图像
img_src = cv2.imread('../img_lib/img_0.jpg', 1)
cv2.imshow('src', img_src)
img_info = img_src.shape
height = img_info[0]
width = img_info[1]
#定义模板
dst_color = np.zeros((height, width, 3), np.uint8)
dst_gray = np.zeros((height, width, 1), np.uint8)

#tf基本运算
def func_0():
    data_0=tf.constant(2.3,name='data_0')
    data_1=tf.Variable(10.0,name='data_1')
    data_add=tf.add(data_0,data_1)
    data_sub=tf.subtract(data_0,data_1)
    data_mul=tf.multiply(data_0,data_1)
    data_div=tf.divide(data_0,data_1)
    data_copy=tf.assign(data_1,data_add)
    with tf.Session() as session:
        init_opra=tf.global_variables_initializer()
        session.run(init_opra)
        print(session.run(data_0))
        print(session.run(data_1))
        print(session.run(data_add))
        print(session.run(data_sub))
        print(session.run(data_mul))
        print(session.run(data_div))
        print(session.run(data_copy))
        print(data_copy.eval())
        print(tf.get_default_session().run(data_copy))
#tf矩阵操作
def func_1 ():
    data_2 = tf.constant([[1,2,3],[4,5,6]])
    with tf.Session() as session:
        print(session.run(data_2))
        print(session.run(data_2[0]))
        print(session.run(data_2)[:,0])
        print(session.run(data_2[1,1]))
#矩阵运算
def func_2():
    data_0 = tf.constant([[1, 2, 3], [4, 5, 6]])
    data_1 = tf.constant([[1, 2, 3], [4, 5, 6],[7,8,9]])
    data_2 = tf.constant([[1, 2], [4, 5]])
    data_3 = tf.constant([[1,2,3],[4,5,6]])

    with tf.Session() as session:
        #print(session.run(tf.add(data_3,data_2)))
        print(session.run(tf.multiply(data_0,data_3)))
        print(session.run(tf.matmul(data_0,data_1)))
#np数据定义
def func_3():
    data_0=np.array([[1,2,3]])
    data_1=np.array([[1,2,3],[4,5,6]])
    data_2=np.array([[7,8,9],[1,2,3]])
    data_3=np.multiply(data_0,data_1)
    data_5=np.array([[1],[2],[3]])
    #data_3=np.multiarray()
    data_4=np.matmul(data_0,data_5)
    data_6=np.add(data_1,data_2)
    data_7=np.zeros([2,3])
    data_8=np.ones([3,5])
    print(data_3,data_4,data_6,data_7,data_8,data_1+data_2)
#绘制图形
def func_4():
    # data_0=np.array([1,2,3,4,5])
    # data_1=np.array([6,3,5,2,6])
    # mpl.plot(data_0,data_1,'r',lw=10)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([3, 5, 7, 6, 2, 6, 10, 15])
    #plt.plot(x, y, 'r')  # 折线 1 x 2 y 3 color
    plt.plot(x, y, 'g', lw=10)  # 4 line w
    # 折线 饼状 柱状
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([13, 25, 17, 36, 21, 16, 10, 15])
    plt.bar(x, y, 0.2, alpha=1, color='b')  # 5 color 4 透明度 3 0.9
    plt.show()
#tf模拟预测数据走势
def func_5():
    end_price=np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2919.08])
    start_price=np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2877.40])
    plt.figure()
    normal_date = np.zeros([15, 1])
    normal_end_price = np.zeros([15, 1])
    for i in range(0,15):
        single_date=np.zeros([2])
        single_date[0]=i
        single_date[1]=i
        single_price=np.zeros([2])
        single_price[0]=start_price[i]
        single_price[1]=end_price[i]
        if end_price[i] > start_price[i]:
            plt.plot(single_date, single_price, 'r', lw=8)
        else:
            plt.plot(single_date, single_price, 'g', lw=8)
        normal_date[i, 0] = i / 14.0
        normal_end_price[i, 0] = end_price[i] / 3000.0
    x=tf.placeholder(tf.float32,shape=[None,1])
    y=tf.placeholder(tf.float32,shape=[None,1])
    w_0=tf.Variable(tf.random_uniform([1,10],0,1))
    b_0=tf.Variable(tf.zeros([1,10]))
    r_0=tf.matmul(x,w_0)+b_0
    layer_0=tf.nn.relu(r_0)
    w_1 = tf.Variable(tf.random_uniform([10, 1], 0, 1))
    b_1 = tf.Variable(tf.zeros([15, 1]))
    r_1 = tf.matmul(layer_0,w_1) + b_1
    layer_1=tf.nn.relu(r_1)
    train_loss=tf.reduce_mean(tf.square(y-layer_1))
    train_step= tf.train.GradientDescentOptimizer(0.1).minimize(train_loss)
    with tf.Session() as session:
        init_opra=tf.global_variables_initializer()
        session.run(init_opra)
        for i in range(0,1000000):
            session.run(train_step,feed_dict={x:normal_date,y:normal_end_price})
        pred_price=session.run(layer_1,feed_dict={x:normal_date})
        predPrice = np.zeros([15, 1])
        print("===预测====")
        print(pred_price)
        for i in range(0, 15):
            predPrice[i, 0] = (pred_price )[i, 0]* 3000
        plt.plot(np.linspace(1,15,15), predPrice, 'b', lw=1)
    plt.show()
#tf模拟预测数据走势
def func_7():
    date = np.linspace(1, 15, 15)
    endPrice = np.array(
        [2511.90, 2538.26, 2510.68, 2591.66, 2732.98, 2701.69, 2701.29, 2678.67, 2726.50, 2681.50, 2739.17, 2715.07,
         2823.58, 2864.90, 2919.08]
    )
    beginPrice = np.array(
        [2438.71, 2500.88, 2534.95, 2512.52, 2594.04, 2743.26, 2697.47, 2695.24, 2678.23, 2722.13, 2674.93, 2744.13,
         2717.46, 2832.73, 2877.40])
    plt.figure()
    for i in range(0, 15):
        # 1 柱状图
        dateOne = np.zeros([2])
        dateOne[0] = i;
        dateOne[1] = i;
        priceOne = np.zeros([2])
        priceOne[0] = beginPrice[i]
        priceOne[1] = endPrice[i]
        if endPrice[i] > beginPrice[i]:
            plt.plot(dateOne, priceOne, 'r', lw=8)
        else:
            plt.plot(dateOne, priceOne, 'g', lw=8)
    # plt.show()
    # A(15x1)*w1(1x10)+b1(1*10) = B(15x10)
    # B(15x10)*w2(10x1)+b2(15x1) = C(15x1)
    # 1 A B C
    dateNormal = np.zeros([15, 1])
    priceNormal = np.zeros([15, 1])
    for i in range(0, 15):
        dateNormal[i, 0] = i / 14.0;
        priceNormal[i, 0] = endPrice[i] / 3000.0;
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])
    # B
    w1 = tf.Variable(tf.random_uniform([1, 10], 0, 1))
    b1 = tf.Variable(tf.zeros([1, 10]))
    wb1 = tf.matmul(x, w1) + b1
    layer1 = tf.nn.relu(wb1)  # 激励函数
    # C
    w2 = tf.Variable(tf.random_uniform([10, 1], 0, 1))
    b2 = tf.Variable(tf.zeros([15, 1]))
    wb2 = tf.matmul(layer1, w2) + b2
    layer2 = tf.nn.relu(wb2)
    loss = tf.reduce_mean(tf.square(y - layer2))  # y 真实 layer2 计算
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 10000):
            sess.run(train_step, feed_dict={x: dateNormal, y: priceNormal})
        # w1w2 b1b2  A + wb -->layer2
        pred = sess.run(layer2, feed_dict={x: dateNormal})
        predPrice = np.zeros([15, 1])
        for i in range(0, 15):
            predPrice[i, 0] = (pred * 3000)[i, 0]
        plt.plot(date, predPrice, 'b', lw=1)
        print(predPrice)
    plt.show()
