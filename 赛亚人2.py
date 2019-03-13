import cv2
import math
import dlib
from PIL import Image
import random
import numpy as np

#电弧图片定义
lightlist = ["l1.png","l2.png","l3.png","l4.png"]
rightlist = ["r1.png","r2.png","r3.png","r4.png"]
toplist = ["t1.png","t2.png","t3.png","t4.png"]

#opencv启用摄像头
cap = cv2.VideoCapture(0)

#dlib面部识别模块相关
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#打开金色头发图片
adding = Image.open('shine.png')
#将窗口定义为可调节大小
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
while True:
    #将摄像头抓取到的结果进行赋值
    _, frame = cap.read()
    #如果帧率不够，可以缩小图片
    frame = cv2.resize(frame,(int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_CUBIC)
    im = Image.fromarray(frame[:,:,::-1])  # 切换RGB格式

    #在摄像头抓取的数据中进行面部识别
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        #获取面部模式
        landmarks = predictor(gray, face)
        #定位面部左上角点坐标
        x1,y1 = landmarks.part(0).x, landmarks.part(0).y
        # 定位面部右上角点坐标
        x2, y2 = landmarks.part(16).x, landmarks.part(16).y
        #计算面部宽度
        d = math.sqrt((x2-x1)**2+(y2-y1)**2)
        #根据面部宽度计算金色头发尺寸
        size = int(d / 236 * 439)
        #对头发图片缩放
        resized = adding.resize((size,size))
        #在合适位置添加头发图片
        im.paste(resized,(int(x1-d*86/236),int(y1-d*394/236)),resized)

        #电弧尺寸
        lightsize = int(d/2)
        #随机获取电弧图片
        ligntning = Image.open(lightlist[random.randint(0, 3)])
        #对电弧进行缩放
        relight = ligntning.resize((lightsize, lightsize))
        #找到合适位置添加电弧图片
        im.paste(relight,(int(x1-d*60/236),int(y1-d*380/236)),relight)

        #定义另一处添加电弧
        ligntning1 = Image.open(lightlist[random.randint(0, 3)])
        relight1 = ligntning1.resize((lightsize, lightsize))
        im.paste(relight1,(int(x1-d*150/236),int(y1-d*200/236)),relight1)

        # 定义另一处添加电弧
        ligntning2 = Image.open(toplist[random.randint(0, 3)])
        relight2 = ligntning2.resize((lightsize, lightsize))
        im.paste(relight2,(int(x1+d*100/236),int(y1-d*450/236)),relight2)

        # 定义另一处添加电弧
        ligntning3 = Image.open(rightlist[random.randint(0, 3)])
        relight3 = ligntning3.resize((lightsize, lightsize))
        im.paste(relight3,(int(x1+d*280/236),int(y1-d*120/236)),relight3)

    #将图片展示在窗口中
    cv2.imshow("Frame", np.array(im)[:,:,::-1]) # 切换RGB格式

    key = cv2.waitKey(1)
    #按ESC键退出摄像头视频
    if key==27:
        break

#退出摄像头、关闭窗口
cap.release()
cv2.destroyAllWindows()