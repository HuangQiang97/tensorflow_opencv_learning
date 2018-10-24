#!/usr/bin/env python
# encoding: utf-8

'''

@author: HuangQiang
@contact: huangqiang97@yahoo.com
@project:
@file: basic_operation.py
@time: 2018/10/14 1:03
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
img_src = cv2.imread('../img_lib/lion.png',1)
cv2.imshow('src', img_src)
img_info = img_src.shape
height = img_info[0]
width = img_info[1]

dst_color = np.zeros((height, width, 3), np.uint8)
dst_gray = np.zeros((height, width, 1), np.uint8)


def func_21_0():
    imgInfo = img_src.shape
    height = imgInfo[0]
    width = imgInfo[1]
    cv2.imshow('src', img_src)
    # sobel 1 算子模版 2 图片卷积 3 阈值判决
    # [1 2 1          [ 1 0 -1
    #  0 0 0            2 0 -2
    # -1 -2 -1 ]       1 0 -1 ]

    # [1 2 3 4] [a b c d] a*1+b*2+c*3+d*4 = dst
    # sqrt(a*a+b*b) = f>th
    gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    dst = np.zeros((height, width, 1), np.uint8)
    for i in range(0, height - 2):
        for j in range(0, width - 2):
            gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 - gray[i + 2, j] * 1 - gray[
                i + 2, j + 1] * 2 - gray[i + 2, j + 2] * 1
            gx = gray[i, j] + gray[i + 1, j] * 2 + gray[i + 2, j] - gray[i, j + 2] - gray[i + 1, j + 2] * 2 - gray[
                i + 2, j + 2]
            grad = math.sqrt(gx * gx + gy * gy)
            if grad > 50:
                dst[i, j] = 255
            else:
                dst[i, j] = 0
    cv2.imshow('dst', dst)
    cv2.imwrite('lion_0.jpg',dst,[cv2.IMWRITE_JPEG_QUALITY ,100])
    cv2.waitKey(0)
func_21_0()

src_img = cv2.imread('../img_lib/image_0.jpg', 1)
def show_img(dst):
    cv2.imshow('dst',dst_color)
def read_img():
    cv2.imshow('img_0',src_img)
    cv2.waitKey(0)
def write():
    cv2.imwrite('../img_out/img_0_copy_0.jpg',src_img)
    cv2.imwrite('../img_out/img_0_copy_1.png',src_img)
    cv2.imwrite('../img_out/img_0_copy_2.jpg',src_img,[cv2.IMWRITE_JPEG_QUALITY,0])
    cv2.imwrite('../img_out/img_0_copy_3.jpg',src_img,[cv2.IMWRITE_JPEG_QUALITY,100])
    cv2.imwrite('../img_out/img_0_copy_4.png',src_img,[cv2.IMWRITE_PNG_COMPRESSION,0])
    cv2.imwrite('../img_out/img_0_copy_5.png',src_img,[cv2.IMWRITE_PNG_COMPRESSION,9])

def write_pixl():
    for i in range(10,110):
        src_img[i,100]=(255,0,0)
    cv2.imwrite('line.jpg',src_img)
#cv数据读取，写出。
def func_6():
    img_src=cv2.imread('../img_lib/img_0.jpg',1)
    cv2.imshow('src',img_src)
    img_info=img_src.shape
    print('属性',img_info)
    dst_0=cv2.resize(img_src,(int(img_info[0]*0.5),int(img_info[1]*0.8)))
    cv2.imshow('0.5*0.8',dst_0)
    cv2.waitKey(0)
#图像放大
def func_8(height_multiple,width_multiple):
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    dst_height=(int(img_info[0]*height_multiple))
    dst_width=(int(img_info[1]*width_multiple))

    dst_0=np.zeros((dst_height,dst_width,3),np.uint8)
    for i in range(dst_height):
        for j in range(dst_width):
            dst_0[i][j]=img_src[int(i/height_multiple),int(j/width_multiple)]
    cv2.imshow('dst_img',dst_0)
    cv2.waitKey(0)
#图像剪裁
def func_9(a,b,c,d):
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    dst_img=img_src[a:b,c:d]
    cv2.imshow('det_img',dst_img)
    cv2.waitKey(0)
#图像移位
def func_10():
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    mat_shift=np.float32([[1,0,100],[0,1,200]])
    dst_0=cv2.warpAffine(img_src,mat_shift,(img_info[0],img_info[1]))
    cv2.imshow('dst',dst_0)
    cv2.waitKey(0)

#图像映射反转
def func_11():
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    height=img_info[0]
    width=img_info[1]
    dst_img=np.zeros((height*2,width,3),np.uint8)
    for i in range(height):
        for j in range(width):
            dst_img[i][j]=img_src[i][j]
            dst_img[2*height-i-1][j]=img_src[i][j]
    cv2.imshow('dst',dst_img)
    cv2.waitKey(2000)
    cv2.imwrite('dst_0.jpg',dst_img)
#图像剪裁。
def func_12():
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    mat_shift = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
    dst_0 = cv2.warpAffine(img_src, mat_shift, (img_info[0], img_info[1]))
    cv2.imshow('dst', dst_0)
    cv2.waitKey(1000)
#图像仿射
def func_13():
    img_src = cv2.imread('../img_lib/img_0.jpg', 1)
    cv2.imshow('src', img_src)
    img_info = img_src.shape
    height=img_info[0]
    width=img_info[1]
    mat_src = np.float32([[0, 0], [0,height-1],[ width-1,0]])
    mat_dst = np.float32([[50, 50], [ 300,height-200], [ width-300,100]])
    mat_affine=cv2.getAffineTransform(mat_src,mat_dst)
    dst_0 = cv2.warpAffine(img_src,mat_affine, (img_info[1], img_info[0]))
    cv2.imshow('dst', dst_0)
    cv2.waitKey(0)
#图像旋转
def func_14():

    mat_rotate=cv2.getRotationMatrix2D((height/3,width/6),35,0.5)
    dst_0=cv2.warpAffine(img_src,mat_rotate,(height,width))
    cv2.imshow('dst',dst_0)
    cv2.waitKey(0)
#图像灰度变换
def func_15():
    dst_0=cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)
    dst_1=np.zeros((height,width,1),np.uint8)
    dst_2 = np.zeros((height, width, 1), np.uint8)
    dst_3 = np.zeros((height, width, 1), np.uint8)
    dst_4 = np.zeros((height, width, 1), np.uint8)
    dst_5 = np.zeros((height, width, 3), np.uint8)



    for  i in range(height):
        for j in range(width):
            (b,g,r)=(img_src[i][j])
            b=int(b)
            g=int(g)
            r=int(r)
            dst_1[i][j]=np.uint8((int(r)+int(g)+int(b))/3)
            dst_5[i][j]=img_src[i][j]

            dst_2[i][j]=np.uint8(int(r)*0.299+int(g)*0.587+int(b)*0.114)
            dst_3[i][j]=np.uint8((r+2*g+b)/4)
            dst_4[i][j]=np.uint8((r+(g<<1)+b)>>2)
    (b)=dst_1[1][2]
    print(b)
    cv2.imshow('dst_0',dst_0)
    cv2.imshow('dst_1',dst_1)
    cv2.imshow('dst_2',dst_2)
    cv2.imshow('dst_3',dst_3)
    cv2.imshow("dst_4",dst_4)
    cv2.imshow('dst_5',dst_5)
    cv2.waitKey(0)
#颜色反转
def func_16():
    dst_0=np.zeros((height,width,3),np.uint8)
    for i in range(height):
        for j in range(width):
            (b,g,r)=img_src[i,j]
            dst_0[i,j]=(255-b,255-g,255-r)

    cv2.imshow('src',img_src)
    cv2.imshow('dst',dst_0)
    cv2.waitKey(0)
#马赛克
def func_17():
    dst_0=np.zeros((height,width,3),np.uint8)
    for i in range(0,height-10,10):
        for j in range(0,width-10,10):
            for m in range(i,i+10):
                for n in range(j,j+10):
                    dst_0[m,n]=img_src[i,j]
    for i in range(0,height-10,10):
        for m in range(i,i+10):
            for n in range(width-10,width):
                dst_0[m,n]=img_src[i,width-10]
    for j in range(0,width-20,10):
        for m in range(height-10,height):
            for n in range(j,j+10):
                dst_0[m,n]=img_src[height-10,j]

    cv2.imshow('dst',dst_0)
    cv2.imshow('src',img_src)
    cv2.waitKey(0)
#毛玻璃
def func_18():
    dst_0=np.zeros((height,width,3),np.uint8)
    for i in  range(0,height-10):
        for j in range(0,width-10):
            random_number=int((random.random())*10)
            dst_0[i][j]=img_src[i+random_number][j+random_number]

    cv2.imshow('src',img_src)
    cv2.imshow('dst',dst_0)
    cv2.waitKey(0)
#图像叠加
def func_19():
    img_src_0=cv2.imread('../img_lib/img_1.jpg')
    img_part_0=img_src[000:300,00:300]
    img_part_1=img_src_0[00:300,00:300]
    dst_0=cv2.addWeighted(img_part_0,0.5,img_part_1,0.5,0)
    cv2.imshow('dst', dst_0)
    cv2.waitKey(0)
#高斯滤波，边缘检测
def func_20():
    img_gray=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    pre_img=cv2.GaussianBlur(img_gray,(3,3),0)
    dst_gray=cv2.Canny(pre_img,20,30)
    cv2.imshow('dst',dst_gray)

    cv2.waitKey(0)
#边缘检测

def func_21():
    gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    for i in range(0,height-2):
        for j in range(0,width-2):
            y=(gary_img[i,j]*1+gary_img[i,j+1]*2+gary_img[i,j+2]*1-gary_img[i+2,j]*1-2*gary_img[i+2,j+1]-gary_img[i+2,j+2]*1)
            x=(gary_img[i,j]*1-gary_img[i,j+2]*1+2*gary_img[i+1,j]-2*gary_img[i+1,j+2]+gary_img[i+2,j]*1-gary_img[i+2,j+2]*1)
            grad= math.sqrt(x**2+y**2)
            if grad>50:
                dst_gray[i,j]=255
            else:dst_gray[i,j]=0
    cv2.imshow('dst',dst_gray)
    cv2.imwrite('flower.jpg', dst_gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.waitKey(0)
#边缘检测

#浮雕特效
def func_22():
    img_gray=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    for i in range(0,height):
        for j in range(0,width-1):
            temp_pixl=int(img_gray[i][j])-int(img_gray[i][j+1])+150
            temp_pixl=temp_pixl if temp_pixl<=255 else 255
            temp_pixl=temp_pixl if temp_pixl>=0 else 0
            dst_gray[i][j]=temp_pixl
    cv2.imshow('dst',dst_gray)
    cv2.waitKey(0)
#颜色映射
def func_23():
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r)=img_src[i,j]
            b=(b*1.5)
            b=b if b<=255 else 255
            g=g if g<=255 else 255

            dst_color[i][j]=(b,g,r)
    cv2.imshow('dst',dst_color)
    cv2.waitKey(0)
#油画特效
def func_24():
    img_gary=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    for i in range(4,height-4):
        for j in range(4,width-4):
            color_array=np.zeros(8,np.uint8)
            for m in range(-4+i,4+i):
                for n in range(-4+j,4+j):
                    color_step=int(img_gary[m,n]/32)
                    color_array[color_step]=color_array[color_step]+1
            currentMax = color_array[0]
            l = 0
            for k in range(0, 8):
                if currentMax < color_array[k]:
                    currentMax = color_array[k]
                    l = k
           # step_index=color_array.tolist().index(max(color_array))
            for m in range(-4+i, 4+i):
                for n in range(-4+j, 4+j):
                    if int(img_gary[ m][ n] / 32)==l:
                        (b,g,r)=img_src[m,n]
                        break
                break
            dst_color[i][j]=(b,g,r)
    cv2.imshow('dst',dst_color)
    cv2.waitKey(0)
#油画特效
def func_24_0():
    img = cv2.imread('../img_lib/image00.jpg', 1)
    cv2.imshow('src', img)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(4, height - 4):
        for j in range(4, width - 4):
            array1 = np.zeros(8, np.uint8)
            for m in range(-4, 4):
                for n in range(-4, 4):
                    p1 = int(gray[i + m, j + n] / 32)
                    array1[p1] = array1[p1] + 1
            currentMax = array1[0]
            l = 0
            for k in range(0, 8):
                if currentMax < array1[k]:
                    currentMax = array1[k]
                    l = k
            # 简化 均值
            for m in range(-4, 4):
                for n in range(-4, 4):
                    if gray[i + m, j + n] >= (l * 32) and gray[i + m, j + n] <= ((l + 1) * 32):
                        (b, g, r) = img[i + m, j + n]
            dst[i, j] = (b, g, r)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
def draw_line():
    cv2.line(dst_color,(100,200),(300,500),(0,0,255),8,cv2.LINE_AA)
    cv2.imshow('dar',dst_color)
    print(dst_color[200,100])
    cv2.waitKey(0)
def draw_rectangle():
    cv2.rectangle(dst_color,(100,200),(400,500),(255,0,0),-1)
    show_img(dst_color)
def draw_cirle():
    cv2.circle(dst_color,(100,100),(100),(0,0,255),2)
    show_img(dst_color)
def drow_ellipse():
    cv2.ellipse(dst_color,(200,300),(150,100),30,0,180,(0,0,255),-1)
    show_img(dst_color)
def draw_polygon():
    points=np.array([[100,200],[200,400],[400,500]],np.uint8)
    points=points.reshape((-1,1,2))
    cv2.polylines(dst_color,points,True,(0,0,255))
    show_img(dst_color)
def draw_words():
    #cv2.rectangle(img_src, (200, 100), (500, 400), (0, 255, 0), 3)
    #cv2.putText(img_src, 'this is flow', (100, 300),cv2.FONT_HERSHEY_SIMPLEX , 1, (200, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(img_src,'this is a text',(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2,cv2.LINE_AA)
    cv2.imshow('wq',img_src)
    cv2.waitKey(0)
    #show_img(img_src)
# img = img_src
# font = cv2.FONT_HERSHEY_SIMPLEX
# #cv2.rectangle(img,(200,100),(500,400),(0,255,0),3)
# # 1 dst 2 文字内容 3 坐标 4 5 字体大小 6 color 7 粗细 8 line type
# cv2.putText(img,'this is flow',(100,300),font,1,(200,100,255),2,cv2.LINE_AA)
# cv2.imshow('src',img)
# cv2.waitKey(0)
def draw_hist(channel_src,channel_type):
    bar_color=(255,255,255)
    window_name=''
    if  channel_type==31:
        bar_color=(255,0,0)
        window_name='blue_hist'
    elif channel_type==32:
        bar_color=(0,255,0)
        window_name='green_hist'
    elif channel_type==33:
        bar_color=(0,0,255)
        window_name=('red_hist')
    color_host=cv2.calcHist([channel_src],[0],None,[256],[0.0,255.0])
    min_value,max_value,min_index ,max_index=cv2.minMaxLoc(color_host)
    hist_img=np.zeros((256,256,3),np.uint8)
    for single_hist in range(256):
        intenNormal=int(color_host[single_hist]*256/max_value)
        cv2.line(hist_img,(single_hist,256),(single_hist,256-intenNormal),bar_color)
    cv2.imshow(window_name,hist_img)
def draw_hist_control():
    color_channels=cv2.split(img_src)
    for i in range(3):
        draw_hist(color_channels[i],31+i)
    cv2.waitKey(0)
def gray_hist_squalize():
    gray_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    dst=cv2.equalizeHist(gray_img)
    color_dst=cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
    cv2.imshow('dst',dst)
    cv2.imshow('color_dat',color_dst)
    cv2.imshow('src',img_src)
    cv2.imshow('gary',gray_img)
    cv2.waitKey(0)
def BGR_hist_equalize():
    img=cv2.imread('../img_lib/image0.jpg',1)
    cv2.imshow('src', img)
    (b,g,r)=cv2.split(img)
    bH=cv2.equalizeHist(b)
    gH=cv2.equalizeHist(g)
    rH=cv2.equalizeHist(r)
    result=cv2.merge((bH , gH, rH))
    cv2.imshow('dst',result)
    cv2.waitKey(0)

    # img = cv2.imread('../img_lib/image0.jpg', 1)
    # cv2.imshow('src', img)
    # (b, g, r) = cv2.split(img)  # 通道分解
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # result = cv2.merge((bH, gH, rH))  # 通道合成
    # cv2.imshow('dst', result)
    # cv2.waitKey(0)
def YUN_equalize():
    yun_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2YCrCb)
    channel_yun=cv2.split(yun_img)
    channel_yun[0]=cv2.equalizeHist(channel_yun[0])
    channels=cv2.merge(channel_yun)
    result=cv2.cvtColor(channels,cv2.COLOR_YCrCb2BGR)
    cv2.imshow('dst',result)
    cv2.imshow('src',img_src)
    cv2.waitKey(0)
def repair_img():
    for i in range(100,200):
        img_src[i,300-1]=(255,255,255)
        img_src[i,300]=(255,255,255)
        img_src[i,300+1]=(255,255,255)
    for j in range(250,350):
        img_src[150-1,j]=(255,255,255)
        img_src[150,j]=(255,255,255)
        img_src[150+1,j]=(255,255,255)
    cv2.imshow('damaged',img_src)
    repair_mask=np.zeros((height,width,1),np.uint8)
    for i in range(100,200):
        repair_mask[i,300-1]=(255)
        repair_mask[i,300]=(255)
        repair_mask[i,300+1]=(255)
    for j in range(250,350):
        repair_mask[150-1,j]=(255)
        repair_mask[150,j]=(255)
        repair_mask[150+1,j]=(255)
    cv2.imshow('repair_mask',repair_mask)
    repaired_img=cv2.inpaint(img_src,repair_mask,2,cv2.INPAINT_TELEA)
    cv2.imshow('repaired_img',repaired_img)
    cv2.waitKey(0)
def gary_hist_0():
    gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    pixl_count=np.zeros(256,np.float)
    for i in range(height):
        for j in range(width):
            pixl_index=int(gary_img[i,j])
            pixl_count[pixl_index]=pixl_count[pixl_index]+1
    for i  in range(256):
        pixl_count[i]=pixl_count[i]/(width*height)
    plt.figure()
    x=np.linspace(0,255,256)
    #print(pixl_count.__len__(),x.__len__())
    plt.bar(x,pixl_count,color='r',alpha=1)
    plt.show()
    cv2.waitKey(0)

    # img = cv2.imread('../img_lib/image0.jpg', 1)
    # imgInfo = img.shape
    # height = imgInfo[0]
    # width = imgInfo[1]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # count = np.zeros(256, np.float)
    # for i in range(0, height):
    #     for j in range(0, width):
    #         pixel = gray[i, j]
    #         index = int(pixel)
    #         count[index] = count[index] + 1
    # for i in range(0, 255):
    #     count[i] = count[i] / (height * width)
    # x = np.linspace(0, 255, 256)
    # y = count
    # plt.bar(x, y, 0.9, alpha=1, color='b')
    # plt.show()
    # cv2.waitKey(0)

def color_hist_0():
    blue_count=np.zeros(256,np.float32)
    green_count=np.zeros(256,np.float32)
    red_count=np.zeros(256,np.float32)
    for i in range(height):
        for j in range(width):
            (b,g,r)=(img_src[i][j])
            blue_count[b]=blue_count[b]+1
            green_count[g]=green_count[g]+1
            red_count[r]=red_count[r]+1
    for i in range(256):
        blue_count[i] = blue_count[i] /(width*height)
        green_count[i] = green_count[i] /(width*height)
        red_count[i] = red_count[i] /(width*height)
    x=np.linspace(0,255,256)
    plt.figure()
    plt.bar(x,blue_count,color='b',alpha=1)
    plt.figure()
    plt.bar(x, green_count, color='g', alpha=1)
    plt.figure()
    plt.bar(x,red_count,color='r',alpha=1)
    plt.show()
    cv2.waitKey(0)
def gray_equalize():
    gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    cv2.imshow('src',gary_img)
    pixel_count=np.zeros(256,np.float32)
    for i in range(height):
        for j in range(width):
            pixel_index=int(gary_img[i,j])
            pixel_count[pixel_index]=pixel_count[pixel_index]+1
    total_pixel=width*height
    probability_sum=np.float32(0)
    pixel_map=np.zeros(256,np.uint16)
    for i in range(256):
        pixel_count[i]=pixel_count[i]/total_pixel
        probability_sum=probability_sum+pixel_count[i]
        pixel_count[i]=probability_sum
       # print(probability_sum)
        pixel_map[i]=np.uint16(pixel_count[i]*256)
    for i in range(height):
        for j in range(width):
            gary_img[i,j]=pixel_map[gary_img[i,j]]
    cv2.imshow('dst',gary_img)
    cv2.waitKey(0)
def color_equalize():
    cv2.imshow('src',img_src)
    blue_count=np.zeros(256,np.float32)
    green_count=np.zeros(256,np.float32)
    red_count=np.zeros(256,np.float32)
    for i in range(height):
        for j in range(width):
            (b,g,r)=img_src[i,j]
            blue_count[b]=blue_count[b]+1
            green_count[g]=green_count[g]+1
            red_count[r]=red_count[r]+1
    pixel_total=height*width
    blue_prob_sum=np.float32(0)
    green_prob_sum=np.float32(0)
    red_prob_sum=np.float32(0)
    blue_map=np.zeros(256,np.float32)
    green_map=np.zeros(256,np.float32)
    red_map=np.zeros(256,np.float32)
    for i in range(256):
        blue_count[i] = blue_count[i] /pixel_total
        green_count[i] = green_count[i] /pixel_total
        red_count[i] = red_count[i]/pixel_total
        blue_prob_sum = blue_prob_sum+blue_count[i]
        green_prob_sum = green_prob_sum+green_count[i]
        red_prob_sum = red_prob_sum+red_count[i]
        blue_map[i]=np.uint16(blue_prob_sum*256)
        green_map[i]=np.uint16(green_prob_sum*256)
        red_map[i]=np.uint16(red_prob_sum*256)

    for i in range(height):
        for j in range(width):
            (b,g,r)=img_src[i,j]
            dst_color[i][j]=(blue_map[b],green_map[g],red_map[r])
    cv2.imshow('dst',dst_color)
    cv2.waitKey(0)
def brightness_enhancement():
    cv2.imshow('src',img_src)
    for i in range(height):
        for j in range(width):
            (b,g,r)=img_src[i,j]
            b=int(b*1.3+10)
            g=int(g*1.2+15)
            if b>255:
                b=255
            if g>255:
                g=255
            img_src[i,j]=(b,g,r)
    cv2.imshow('dst',img_src)
    cv2.waitKey(0)
def bilatera_filter():
    cv2.imshow('src',img_src)
    dst=cv2.bilateralFilter(img_src,15,35,35)
    cv2.imshow('dst',dst_color)
    cv2.waitKey(0)
def gary_midle_filter():
    gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    cv2.imshow('src',gary_img)
    for i in range(1,height-1):
        for j in range(1,width-1):
            k=0
            around_pixel=np.zeros(9,np.uint8)
            for m in range(i-1,i+2):
                for n in range(j-1,j+2):
                    around_pixel[k]= gary_img[m,n]
                    k=k+1
            sorted_array=sorted(around_pixel)
            gary_img[i,j]=sorted_array[4]
    cv2.imshow('dst',gary_img)
    cv2.waitKey(0)
def gary_avg_filter():
    gary_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    cv2.imshow('src', gary_img)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            pixel_sum = 0
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    pixel_sum=pixel_sum+gary_img[m,n]
            gary_img[i, j] = int(pixel_sum/9)
    cv2.imshow('dst', gary_img)
    cv2.waitKey(0)
def color_midle_filter():
    cv2.imshow('src',img_src)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            k = 0
            around_b_pixel = np.zeros(9, np.uint8)
            around_g_pixel = np.zeros(9, np.uint8)
            around_r_pixel = np.zeros(9, np.uint8)
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    (around_b_pixel[k],around_g_pixel[k],around_r_pixel[k])=img_src[m,n]
                    k = k + 1
            sorted_b_array = sorted(around_b_pixel)
            sorted_g_array = sorted(around_g_pixel)
            sorted_r_array = sorted(around_r_pixel)
            dst_color[i, j] = (sorted_b_array[4],sorted_g_array[4],sorted_r_array[4])
    cv2.imshow('dst',dst_color )
    cv2.waitKey(0)
def color_avg_filter():
    cv2.imshow('src',img_src)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            pixel_b_sum = 0
            pixel_g_sum = 0
            pixel_r_sum = 0
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    (b,g,r)=img_src[m,n]
                    pixel_b_sum=pixel_b_sum+b
                    pixel_g_sum=pixel_g_sum+g
                    pixel_r_sum=pixel_r_sum+r
            dst_color[i, j] = (pixel_b_sum/9,pixel_g_sum/9,pixel_r_sum/9)
    cv2.imshow('dst', dst_color)
    cv2.waitKey(0)

# color_midle_filter()
#color_avg_filter()



