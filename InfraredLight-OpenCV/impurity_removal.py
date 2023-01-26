#-*-coding:utf-8-*-

import cv2
import time
import numpy as np


class ImpurityRemoval:
    def interference_detection(self, frame_lwpCV) -> bool:     # 返回True代表为干扰，返回False代表不为干扰
        h, w = frame_lwpCV.shape[:2] # template_gray 为灰度图
        m = np.reshape(frame_lwpCV, [1, w*h])
        mean = m.sum()/(w*h) # 图像平均灰度值
        # print("Gray_Score_is " , mean , "\n")
        # cv2.imshow("interference_detection" , frame_lwpCV)
        # time.sleep(3)
        if mean < 100: # 假设灰度平均值小于100是干扰
            return True
        else:
            return False
