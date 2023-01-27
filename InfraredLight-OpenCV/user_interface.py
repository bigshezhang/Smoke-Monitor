#-*-coding:utf-8-*-

import sys
import cv2

from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLCDNumber
from PyQt6.QtGui import QPixmap
import sys
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import video

selected_video_str = "close_range.mp4"

class UserInterface(QWidget):
    thread = video.ContourDetection()
    def __init__(self):
        global selected_video_str
        super().__init__()
        self.setWindowTitle("OpenCV 2 Qt")
        self.disply_width = 320
        self.display_height = 240
        self.cv_frame_label = QLabel(text = 'cv_frame')
        self.cv_gray_label = QLabel(text = 'cv_gray')
        self.cv_diff_label = QLabel(text = 'cv_diff')
        self.cv_piece_label = QLabel(text = 'cv_piece')
        self.cv_frame_label.resize(480, 480)
        self.cv_gray_label.resize(480, 480)
        self.cv_diff_label.resize(480, 480)
        self.cv_piece_label.setFixedSize(100, 160)

        cv_piece_vbox = QVBoxLayout()
        self.cv_piece_gray_scale_label = QLabel(text = 'Gray Scale: ')
        self.cv_piece_area_scale_label = QLabel(text = 'Area Scale: ')

        cv_piece_vbox.addWidget(self.cv_piece_label)
        cv_piece_vbox.addWidget(self.cv_piece_gray_scale_label)
        cv_piece_vbox.addWidget(self.cv_piece_area_scale_label)

        self.thread = video.ContourDetection()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.piece_pixmap_signal.connect(self.update_piece)
        self.thread.start()

        qbtn_vbox = QVBoxLayout()

        qbtn_close_range = QPushButton('Close Range', self)
        qbtn_close_range.clicked.connect(self.buttonClicked)
        qbtn_close_range.clicked.connect(self.thread.start)
        qbtn_close_range.resize(qbtn_close_range.sizeHint())

        qbtn_far_range = QPushButton('Far Range', self)
        qbtn_far_range.clicked.connect(self.buttonClicked)
        qbtn_far_range.clicked.connect(self.thread.start)
        qbtn_far_range.resize(qbtn_far_range.sizeHint())

        qbtn_interference_nearby = QPushButton('Interference Nearby', self)
        qbtn_interference_nearby.clicked.connect(self.buttonClicked)
        qbtn_interference_nearby.clicked.connect(self.thread.start)
        qbtn_interference_nearby.resize(qbtn_interference_nearby.sizeHint())

        qbtn_interference_faraway = QPushButton('Interference Faraway', self)
        qbtn_interference_faraway.clicked.connect(self.buttonClicked)
        qbtn_interference_faraway.clicked.connect(self.thread.start)
        qbtn_interference_faraway.resize(qbtn_interference_faraway.sizeHint())

        qbtn_change_speed = QPushButton('Change Speed', self)
        qbtn_change_speed.setCheckable(True)
        qbtn_change_speed.clicked[bool].connect(self.thread.change_speed)
        qbtn_change_speed.resize(qbtn_change_speed.sizeHint())

        qbtn_vbox.addWidget(qbtn_close_range)
        qbtn_vbox.addWidget(qbtn_far_range)
        qbtn_vbox.addWidget(qbtn_interference_nearby)
        qbtn_vbox.addWidget(qbtn_interference_faraway)
        qbtn_vbox.addWidget(qbtn_change_speed)

        gray_threshold_sld_title = QLabel('Gray Threshold')
        gray_threshold_sld = QSlider(Qt.Orientation.Horizontal, self)
        gray_threshold_sld.setValue(100)
        gray_threshold_lcd = QLCDNumber(self)
        gray_threshold_lcd.display(100)
        gray_threshold_sld.setMaximum(255)
        gray_hbox = QHBoxLayout()
        gray_hbox.addWidget(gray_threshold_sld_title)
        gray_hbox.addWidget(gray_threshold_sld)
        gray_hbox.addWidget(gray_threshold_lcd)
        gray_threshold_sld.valueChanged.connect(gray_threshold_lcd.display)
        gray_threshold_sld.valueChanged.connect(self.thread.update_gray_threshold)

        area_threshold_sld_title = QLabel('Area Threshold')
        area_threshold_sld = QSlider(Qt.Orientation.Horizontal, self)
        area_threshold_sld.setValue(500)
        area_threshold_lcd = QLCDNumber(self)
        area_threshold_lcd.display(500)
        area_threshold_sld.setMaximum(30000)
        area_hbox = QHBoxLayout()
        area_hbox.addWidget(area_threshold_sld_title)
        area_hbox.addWidget(area_threshold_sld)
        area_hbox.addWidget(area_threshold_lcd)
        area_threshold_sld.valueChanged.connect(area_threshold_lcd.display)
        area_threshold_sld.valueChanged.connect(self.thread.update_area_threshold)


        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(self.cv_frame_label)
        hbox.addWidget(self.cv_gray_label)
        hbox.addWidget(self.cv_diff_label)
        hbox.addLayout(cv_piece_vbox)
        hbox.addLayout(qbtn_vbox)

        vbox.addLayout(hbox)
        vbox.addLayout(gray_hbox)
        vbox.addLayout(area_hbox)

        self.setLayout(vbox)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image(self, cv_frame_img, cv_gray_img, cv_diff_img):
        # print('Update Img')
        qt_cv_frame_img = self.convert_cv_bgr_qt(cv_frame_img)
        self.cv_frame_label.setPixmap(qt_cv_frame_img)

        qt_cv_gray_img = self.convert_cv_gray_qt(cv_gray_img)
        self.cv_gray_label.setPixmap(qt_cv_gray_img)

        qt_cv_diff_img = self.convert_cv_gray_qt(cv_diff_img)
        self.cv_diff_label.setPixmap(qt_cv_diff_img)
    
    @pyqtSlot(np.ndarray, int, int)
    def update_piece(self, cv_piece_img, gray_scale, area_scale):
        qt_cv_piece_img = self.convert_cv_gray_qt(cv_piece_img)
        self.cv_piece_label.setPixmap(qt_cv_piece_img)
        self.cv_piece_gray_scale_label.setText(('Gray Scale: %d' % gray_scale))
        self.cv_piece_area_scale_label.setText(('Area Scale: %d' % area_scale))

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

    def buttonClicked(self):
        global selected_video_str
        sender = self.sender()

        match sender.text():
            case 'Close Range':
                selected_video_str = 'close_range.mp4'
                # print("1")
            case 'Far Range':
                selected_video_str = 'far_range.mp4'
                # print("2")
            case 'Interference Nearby':
                selected_video_str = 'interference_nearby.mp4'
            case 'Interference Faraway':
                selected_video_str = 'interference_faraway.mp4'
            case _:
                selected_video_str = 'interference_faraway.mp4'
            

ui = any
