#-*-coding:utf-8-*-

import cv2
import os

import select_interface
import impurity_removal

class ContourDetection:

  def __init__(self):
    self.__camera = cv2.VideoCapture(os.getcwd() + '/' + select_interface.selected_video_str)    # 在路径下打开文件
    # # 测试用,查看视频size
    # self.__size = (int(__camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #   int(__camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    self.__es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)) #构造了一个特定的9-4矩形内切椭圆，用作卷积核
    self.__background = None
 
  def capture_contours(self):
    print("已调用边缘检测模块，检测的视频为" + select_interface.selected_video_str)
    while True:
      # 读取视频流
      grabbed, frame_lwpCV = self.__camera.read()
      if grabbed == False:  # 若捕获的帧为假，退出程序
        break
      # 对帧进行预处理，先转灰度图，再进行高斯滤波。
      # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
      gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
      gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
 
      # 将第一帧设置为整个输入的背景
      if self.__background is None:
        self.__background = gray_lwpCV
      # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
      # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
      diff = cv2.absdiff(self.__background, gray_lwpCV)
      diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
      diff = cv2.dilate(diff,self.__es, iterations=2) # 形态学膨胀
      diff = cv2.erode(diff, self.__es, iterations=2)  # 形态学腐蚀
 
      # 显示矩形框
      contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
      for c in contours:
        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框，其中xywh格式的坐标代表 左上角坐标(x,y)和宽高(w,h)
        if cv2.contourArea(c) < 1500 \
          or impurity_removal.ImpurityRemoval().interference_detection \
          (frame_lwpCV[y : y + h, x : x + w]): 
            # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
          continue
        cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(diff, (x, y), (x+w, y+h), (255, 255, 255), 2)  # 在差分图像上显示矩形框，颜色为白色(255,255,255)
 
      cv2.imshow('contours', frame_lwpCV)
      cv2.imshow('dis', diff)
      self.__background = gray_lwpCV
  
      key = cv2.waitKey(1) & 0xFF
      # 按'q'健退出循环
      if key == ord('q'):
        break

    self.__camera.release()
    cv2.destroyAllWindows()


