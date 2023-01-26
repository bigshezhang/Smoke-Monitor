#-*-coding:utf-8-*-

import sys
import cv2

from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap
import sys
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import video

class UserInterface(QWidget):
    def receive_from_cv(self, frame_lwpCV: any, gray_lwpCV: any, diff: any, is_ready: bool):
        print('Receive_From_CV')
        self.cv_frame_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2RGB)
        self.cv_gray_lwpCV = cv2.cvtColor(gray_lwpCV, cv2.COLOR_GRAY2RGB) 
        self.cv_diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB) 
        self.is_cv_ready = is_ready

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV 2 Qt")
        self.disply_width = 320
        self.display_height = 240
        self.cv_frame_label = QLabel(text = 'cv_frame')
        self.cv_gray_label = QLabel(text = 'cv_gray')
        self.cv_diff_label = QLabel(text = 'cv_diff')
        self.cv_frame_label.resize(480, 480)
        self.cv_gray_label.resize(480, 480)
        self.cv_diff_label.resize(480, 480)

        hbox = QHBoxLayout()
        hbox.addWidget(self.cv_frame_label)
        hbox.addWidget(self.cv_gray_label)
        hbox.addWidget(self.cv_diff_label)

        self.setLayout(hbox)

        self.thread = video.ContourDetection()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_frame_img, cv_gray_img, cv_diff_img):
        # print('Update Img')
        qt_cv_frame_img = self.convert_cv_bgr_qt(cv_frame_img)
        self.cv_frame_label.setPixmap(qt_cv_frame_img)

        qt_cv_gray_img = self.convert_cv_gray_qt(cv_gray_img)
        self.cv_gray_label.setPixmap(qt_cv_gray_img)
        qt_cv_diff_img = self.convert_cv_gray_qt(cv_diff_img)
        self.cv_diff_label.setPixmap(qt_cv_diff_img)
    
    def convert_cv_bgr_qt(self, cv_img):
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_cv_gray_qt(self, cv_img):
        h, w = cv_img.shape
        bytes_per_line = w
        convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

ui = any
