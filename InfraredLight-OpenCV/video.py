import numpy as np
import cv2
import os

path = os.getcwd()#获取当前路径
print(path)

camera = cv2.VideoCapture(path + '/red.mp4')    #在相对路径下打开文件
# 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))
 
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
 
while True:
  # 读取视频流
  grabbed, frame_lwpCV = camera.read()
  # 对帧进行预处理，先转灰度图，再进行高斯滤波。
  # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
  gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
  gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
 
  # 将第一帧设置为整个输入的背景
  if background is None:
    background = gray_lwpCV
    continue
  # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
  # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
  diff = cv2.absdiff(background, gray_lwpCV)
  diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
  diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
  diff = cv2.erode(diff, es, iterations=2)  # 形态学腐蚀
 
  # 显示矩形框
  contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
  for c in contours:
    if cv2.contourArea(c) < 1500: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
      continue
    (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
    cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
  cv2.imshow('contours', frame_lwpCV)
  cv2.imshow('dis', diff)
  background = gray_lwpCV
  
  key = cv2.waitKey(1) & 0xFF
  # 按'q'健退出循环
  if key == ord('q'):
    break

camera.release()
cv2.destroyAllWindows()
