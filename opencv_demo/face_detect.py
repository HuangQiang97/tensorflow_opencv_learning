#!/usr/bin/env python
# encoding: utf-8

'''

@author: HuangQiang
@contact: huangqiang97@yahoo.com
@project:
@file: face_detect.py
@time: 2018/10/23 0:56
@desc:
@version:

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  cv2
def face_eye_detect():
    face_xml=cv2.CascadeClassifier('../module_lib/haarcascade_frontalface_default.xml')
    eye_xml=cv2.CascadeClassifier('../module_lib/haarcascade_eye.xml')
    profile_face_xml=cv2.CascadeClassifier("../module_lib/haarcascade_profileface.xml")
    video_src = cv2.VideoCapture('../img_lib/gay.mp4')
    while True:
        (ret, frame) = video_src.read()
        img_src = cv2.flip(frame, 1)
        gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
        cv2.imshow('video', img_src)

        # img_src=cv2.imread('../img_lib/lina.jpg',1)
        # cv2.imshow('src',img_src)
        # gary_img=cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gary',gary_img)
        faces=face_xml.detectMultiScale(gary_img,1.5,1)
        profile_faces=profile_face_xml.detectMultiScale(gary_img,1.5,1)
        print('face：',len(faces)+len(profile_faces))
        for (face_start_width,face_start_height,face_width,face_height) in faces:
            cv2.rectangle(img_src,(face_start_width,face_start_height),(face_start_width+face_width,face_start_height+face_height)
                          ,(0.0,255),4)
            cv2.putText(img_src,'',(face_start_width+face_width,face_start_height),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
            gary_face_ori=gary_img[face_start_height:face_start_height+face_height,face_start_width:face_start_width+face_width]
            color_face=img_src[face_start_height:face_start_height+face_height,face_start_width:face_start_width+face_width]
            # eyes=eye_xml.detectMultiScale(gary_face_ori,2,1)
            # print('eye：',len(eyes))
            # for (eye_start_width, eye_start_height,eye_width, eye_height) in eyes:
            #     cv2.rectangle(color_face,(eye_start_width,eye_start_height),(eye_start_width+eye_width,eye_start_height+eye_height)
            #               ,(0.0,255),4)
        for (face_start_width,face_start_height,face_width,face_height) in profile_faces:
            cv2.rectangle(img_src,(face_start_width,face_start_height),(face_start_width+face_width,face_start_height+face_height)
                          ,(0.0,255),4)
            cv2.putText(img_src,'',(face_start_width+face_width,face_start_height),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
            gary_face_ori=gary_img[face_start_height:face_start_height+face_height,face_start_width:face_start_width+face_width]
            color_face=img_src[face_start_height:face_start_height+face_height,face_start_width:face_start_width+face_width]
            # eyes=eye_xml.detectMultiScale(gary_face_ori,2,1)
            # print('eye：',len(eyes))

            # for (eye_start_width, eye_start_height,eye_width, eye_height) in eyes:
            #     cv2.rectangle(color_face,(eye_start_width,eye_start_height),(eye_start_width+eye_width,eye_start_height+eye_height)
            #               ,(0.0,255),4)

        cv2.imshow('dst',img_src)
        if cv2.waitKey(60)&0xffff==27:
            break
    video_src.release()
    cv2.destroyAllWindows()
face_eye_detect()

