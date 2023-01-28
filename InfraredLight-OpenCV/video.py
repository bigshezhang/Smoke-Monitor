#-*-coding:utf-8-*-

import cv2
import os
import time

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import user_interface

class ContourDetection(QThread):    # 在构建可视化软件时，耗费计算资源的线程尽量不占用主线程，本类继承于 QThread
  
  change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)     # 两个信号，用于提醒更新界面中的图片框
  piece_pixmap_signal = pyqtSignal(np.ndarray, int, int)

  def run(self):
    self.__camera = cv2.VideoCapture(os.getcwd() + '/video/' + user_interface.selected_video_str)    # 在路径下打开文件
    self.__es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)) #构造了一个特定的9-4矩形内切椭圆，用作卷积核
    self.__gray_threshold = 100   # 初始化差分图像的干扰滤除中，灰度阈值与面积阈值
    self.__area_threshold = 500
    self.delay = 0.0
    print("已调用边缘检测模块")
    while True:
      
      # 读取视频流
      grabbed, frame_lwpCV = self.__camera.read()
      frame_queue = [frame_lwpCV]
      # 在循环中读取帧
      while True:
        time.sleep(self.delay)
        # 读取当前帧
        grabbed, frame_lwpCV = self.__camera.read()
        # 将当前帧添加到队列中
        frame_queue.append(frame_lwpCV)
        # 如果队列长度大于 5，则移除最早的帧
        if len(frame_queue) > 5:
          frame_queue.pop(0)
        # 获取队列中最后一帧和第一帧
        last_frame = frame_queue[len(frame_queue) - 1]
        first_frame = frame_queue[0]

        try:
          gray_last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
          gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

          # 对帧进行预处理，先转灰度图，再进行高斯滤波。
          # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
          gray_last_frame = cv2.GaussianBlur(gray_last_frame, (21, 21), 0)
          gray_first_frame = cv2.GaussianBlur(gray_first_frame, (21, 21), 0)

          # 使用 OpenCV 进行帧差处理
          diff = cv2.absdiff(gray_last_frame, gray_first_frame)
          # 显示处理结果
          # cv2.imshow("diff", diff)
          # 等待按键
        except:
          print("结束")
        
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
        diff = cv2.dilate(diff,self.__es, iterations=2) # 形态学膨胀
        diff = cv2.erode(diff, self.__es, iterations=2) # 形态学腐蚀
        # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
        for c in contours:
          (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框，其中xywh格式的坐标代表 左上角坐标(x,y)和宽高(w,h)
          m = np.reshape(gray_first_frame[y : y + h, x : x + w], [1, w * h])
          mean = m.sum()/(w * h) # 图像平均灰度值
        
          if w * h < self.__area_threshold or int(mean) < self.__gray_threshold: # 向干扰剔除模块传入一个灰度图，用于判断灰度平均值
            # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
          self.piece_pixmap_signal.emit(gray_first_frame[y : y + h, x : x + w].copy(), int(mean), w * h) # 需使用 .copy() 否则读入缓冲区内容，后续报错
          cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
          cv2.rectangle(diff, (x, y), (x+w, y+h), (255, 255, 255), 2)  # 在差分图像上显示矩形框，颜色为白色(255,255,255)
          cv2.rectangle(gray_first_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)  # 在差分图像上显示矩形框，颜色为白色(255,255,255)
        try:
          self.change_pixmap_signal.emit(frame_lwpCV,gray_first_frame, diff)
        except:
          break

        key = cv2.waitKey(1)
        if key == 27:
          break
      if grabbed == False:  # 若捕获的帧为假，退出程序
        break
     
      key = cv2.waitKey(1) & 0xFF
      # 按'q'健退出循环
      if key == ord('q'):
        break

    self.__camera.release()
    cv2.destroyAllWindows()

  @pyqtSlot(int)
  def update_gray_threshold(self, threshold):   # 分别是更新灰度阈值与面积阈值的插槽
    self.__gray_threshold = threshold

  @pyqtSlot(int)
  def update_area_threshold(self, threshold):
    self.__area_threshold = threshold

  @pyqtSlot(bool)                               # 更新 OpenCV 读入每两帧之间的休眠时间
  def change_speed(self, pressed):
    if pressed:
      self.delay = 0.3
    else:
      self.delay = 0

