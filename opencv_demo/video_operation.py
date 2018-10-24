#!/usr/bin/env python
# encoding: utf-8

'''

@author: HuangQiang
@contact: huangqiang97@yahoo.com
@project:
@file: video_operation.py
@time: 2018/10/21 23:58
@desc:
@version:

'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

def video2img():
    video_src=cv2.VideoCapture('../img_lib/video_0.mp4')
    video_status=video_src.isOpened()
    fps=video_src.get(cv2.CAP_PROP_FPS)
    img_height=int(video_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width=int(video_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(video_status,fps,img_height,img_width)
    k=0
    total_frame=int(video_src.get(cv2.CAP_PROP_FRAME_COUNT))/10
    print(total_frame)
    while k<total_frame:
        (flag,img)=video_src.read()
        # if not float:
        #     break
        # cv2.imshow('video',img)
        # cv2.waitKey(100)
        file_name='../img_out/'+str(k)+'.jpg'
        cv2.imwrite(file_name,img,[cv2.IMWRITE_JPEG_QUALITY,100])
        k=k+1

def get_video():
    capture=cv2.VideoCapture(0)
    while True:
        (ret, frame)=capture.read()
        #img_shape=frame.shape
        #print(img_shape[1],img_shape[0])
        frame=cv2.flip(frame,1)
        cv2.imshow('video',frame)
       # print(frame.shape)
        cv2.waitKey(100)



def img2video():
    src_shape=cv2.imread('../img_out/0.jpg',1).shape
    print(src_shape)
    size=(src_shape[1],src_shape[0])
    video_dst=cv2.VideoWriter('../img_out/out_video_0.mp4',-1,5,size,True)
    k=0
    while k<20:
        file_path='../img_out/'+str(k)+'.jpg'
        img_src=cv2.imread(file_path,1)
        cv2.imshow('video',img_src)
        cv2.waitKey(10)
        video_dst.write(img_src)
        k=k+1
    video_dst.release()
def a():
    capture = cv2.VideoCapture(0)
    k=0
    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # VideoWriter 参数1: 写入对象 参数
    videoWrite = cv2.VideoWriter('video.mp4', -1, 10, size, True)
    while k<5000:
        (ret, frame) = capture.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('video',frame)

          # 写入对象 1 file name# 2 可用编码器 3 帧率 4 size
        videoWrite.write(frame)  # 写入方法 1 jpg dataprint('end!')
        k=k+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    capture.release()
    videoWrite.release()
a()
# img = cv2.imread('../img_lib/img_0.jpg')
# imgInfo = img.shape
# size = (imgInfo[1],imgInfo[0])
# print(size)
# videoWrite = cv2.VideoWriter('2.mp4',-1,5,size)# 写入对象 1 file name
# # 2 编码器 3 帧率 4 size
# for i in range(1,11):
#     fileName = 'image'+str(i)+'.jpg'
#     img = cv2.imread(fileName)
#     videoWrite.write(img)# 写入方法 1 jpg data
# print('end!')

# import numpy as np
#
# capture = cv2.VideoCapture('../img_lib/video_0.mp4',1)
# size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
# out = cv2.VideoWriter('../img_out/001_output.mp4', fourcc, 29.0, size,
#                       False)  # 'False' for 1-ch instead of 3-ch for color
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while (capture.isOpened()):  # while Ture:
#     ret, img = capture.read()
#     if ret == True:
#         fgmask = fgbg.apply(img)
#         out.write(fgmask)
#         #cv2.imshow('img', fgmask)
#
#     # if(cv2.waitKey(27)!=-1):  # observed it will close the imshow window immediately
#     #    break                 # so change to below
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# out.release()
# cv2.destroyAllWindows()




    # img = cv2.imread('../img_out/0.jpg')
    # imgInfo = img.shape  # 宽度和高度信
    # size = (imgInfo[1],imgInfo[0])
    # print(size)  # windows下使用DIVX
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')# VideoWriter 参数1: 写入对象 参数
    # videoWrite = cv2.VideoWriter('pic2video.avi',fourcc,1,size,True)# 写入对象 1 file name# 2 可用编码器 3 帧率 4 size
    # for i in range(1,11):
    #     fileName = 'image' + str(i) + '.jpg'
    #     img = cv2.imread(fileName)
    #     videoWrite.write(img)  # 写入方法 1 jpg dataprint('end!')


