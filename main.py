#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import sys
from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLCDNumber
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

from OpenCV import video
from Yolov5 import yolo

# cv_selected_video_str = "simulation.mp4"  # 用来指定播放哪一段视频的地址字符串

def main():
    app = QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    sys.exit(app.exec())

class UserInterface(QWidget):
    yolo_thread = yolo.YoloDetection()
    cv_thread = video.ContourDetection()   # 此处 OpenCV 图像处理部分被列为子线程，主线程为 GUI 交互界面
    def __init__(self):
        global cv_selected_video_str
        super().__init__()
        self.setWindowTitle("OpenCV 2 Qt")
        self.disply_width = 320
        self.display_height = 240                           # 四个 QLabel 分别为
        self.cv_frame_label = QLabel(text = 'cv_frame')     # 将源视频流输出
        self.cv_gray_label = QLabel(text = 'cv_gray')       # 将灰度处理过的图像输出
        self.cv_diff_label = QLabel(text = 'cv_diff')       # 将差分处理且二值化后的图像输出
        self.cv_piece_label = QLabel(text = 'cv_piece')     # 输出差分图像中每一块轮廓的灰度图，此图的亮度平均值可判断是否为干扰
        self.yolo_label = QLabel(text='yolo_frame')
        self.cv_frame_label.resize(480, 480)                # 设置图像框的大小(此处不太了解，似乎约束力不强)
        self.cv_gray_label.resize(480, 480)
        self.cv_diff_label.resize(480, 480)
        self.cv_piece_label.setFixedSize(100, 160)          # setFixedSize 为强约束
        self.yolo_label.resize(480, 480)

        cv_piece_vbox = QVBoxLayout()                       # 将轮廓图与其平均亮度、区域面积的数据纳入 VBox 中
        self.cv_piece_gray_scale_label = QLabel(text = 'Gray Scale: ')
        self.cv_piece_area_scale_label = QLabel(text = 'Area Scale: ')

        cv_piece_vbox.addWidget(self.cv_piece_label)
        cv_piece_vbox.addWidget(self.cv_piece_gray_scale_label)
        cv_piece_vbox.addWidget(self.cv_piece_area_scale_label)

        self.cv_thread = video.ContourDetection()              # 以子线程模式运行 OpenCV 图像处理
        self.cv_thread.cv_change_pixmap_signal.connect(self.cv_update_image)     # 将发送端(包括图片信息)的信号接入插槽
        self.cv_thread.piece_pixmap_signal.connect(self.cv_update_piece)
        self.yolo_thread.yolo_change_pixmap_signal.connect(self.yolo_update_image)
        self.yolo_thread.yolo_change_status.connect(self.yolo_launched_check)
        self.yolo_thread.start()
        print("运动检测模块等待同步启动中")

        qbtn_vbox = QVBoxLayout()                           # 一堆纵向排列的按钮，用于选择视频输入与改变输入速度，使用 VBox 包裹

        qbtn_close_range = QPushButton('Close Range', self)
        qbtn_close_range.clicked.connect(self.buttonClicked)
        qbtn_close_range.clicked.connect(self.cv_thread.start)
        qbtn_close_range.resize(qbtn_close_range.sizeHint())

        qbtn_far_range = QPushButton('Far Range', self)
        qbtn_far_range.clicked.connect(self.buttonClicked)
        qbtn_far_range.clicked.connect(self.cv_thread.start)
        qbtn_far_range.resize(qbtn_far_range.sizeHint())

        qbtn_interference_nearby = QPushButton('Interference Nearby', self)
        qbtn_interference_nearby.clicked.connect(self.buttonClicked)
        qbtn_interference_nearby.clicked.connect(self.cv_thread.start)
        qbtn_interference_nearby.resize(qbtn_interference_nearby.sizeHint())

        qbtn_interference_faraway = QPushButton('Interference Faraway', self)
        qbtn_interference_faraway.clicked.connect(self.buttonClicked)
        qbtn_interference_faraway.clicked.connect(self.cv_thread.start)
        qbtn_interference_faraway.resize(qbtn_interference_faraway.sizeHint())

        qbtn_change_speed = QPushButton('Change Speed', self)       # 改变 OpenCV 中图像输入速度
        qbtn_change_speed.setCheckable(True)
        qbtn_change_speed.clicked[bool].connect(self.cv_thread.change_speed)
        qbtn_change_speed.resize(qbtn_change_speed.sizeHint())

        qbtn_vbox.addWidget(qbtn_close_range)
        qbtn_vbox.addWidget(qbtn_far_range)
        qbtn_vbox.addWidget(qbtn_interference_nearby)
        qbtn_vbox.addWidget(qbtn_interference_faraway)
        qbtn_vbox.addWidget(qbtn_change_speed)

        gray_threshold_sld_title = QLabel('Gray Threshold')         # 两条滑块，调节差分轮廓的灰度阈值与面积阈值
        gray_threshold_sld = QSlider(Qt.Orientation.Horizontal, self)       # 通过调节阈值了解不同情况下适用的参数(future)
        gray_threshold_lcd = QLCDNumber(self)
        gray_threshold_lcd.display(50)
        gray_threshold_sld.setMaximum(255)
        gray_threshold_sld.setValue(50)
        gray_hbox = QHBoxLayout()
        gray_hbox.addWidget(gray_threshold_sld_title)
        gray_hbox.addWidget(gray_threshold_sld)
        gray_hbox.addWidget(gray_threshold_lcd)
        gray_threshold_sld.valueChanged.connect(gray_threshold_lcd.display)
        gray_threshold_sld.valueChanged.connect(self.cv_thread.update_gray_threshold)

        area_threshold_sld_title = QLabel('Area Threshold')
        area_threshold_sld = QSlider(Qt.Orientation.Horizontal, self)
        area_threshold_lcd = QLCDNumber(self)
        area_threshold_lcd.display(50000)
        area_threshold_sld.setMaximum(30000)
        area_threshold_sld.setValue(5000)
        area_hbox = QHBoxLayout()
        area_hbox.addWidget(area_threshold_sld_title)
        area_hbox.addWidget(area_threshold_sld)
        area_hbox.addWidget(area_threshold_lcd)
        area_threshold_sld.valueChanged.connect(area_threshold_lcd.display)
        area_threshold_sld.valueChanged.connect(self.cv_thread.update_area_threshold)

        skip_frame_sld_title = QLabel('Skip Frame')
        skip_frame_sld = QSlider(Qt.Orientation.Horizontal, self)
        skip_frame_lcd = QLCDNumber(self)
        skip_frame_lcd.display(5)
        skip_frame_sld.setMaximum(5)
        skip_frame_sld.setValue(5)
        skip_frame_hbox = QHBoxLayout()
        skip_frame_hbox.addWidget(skip_frame_sld_title)
        skip_frame_hbox.addWidget(skip_frame_sld)
        skip_frame_hbox.addWidget(skip_frame_lcd)
        skip_frame_sld.valueChanged.connect(skip_frame_lcd.display)
        skip_frame_sld.valueChanged.connect(self.cv_thread.update_skip_frame)

        hbox = QHBoxLayout()
        cv_vbox = QVBoxLayout()

        hbox.addWidget(self.cv_frame_label)
        hbox.addWidget(self.cv_gray_label)
        hbox.addWidget(self.cv_diff_label)
        hbox.addLayout(cv_piece_vbox)
        hbox.addLayout(qbtn_vbox)

        cv_vbox.addLayout(hbox)
        cv_vbox.addLayout(gray_hbox)
        cv_vbox.addLayout(area_hbox)
        cv_vbox.addLayout(skip_frame_hbox)

        hyper_vbox = QVBoxLayout()
        hyper_vbox.addLayout(cv_vbox)
        hyper_vbox.addWidget(self.yolo_label)



        self.setLayout(hyper_vbox)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)       # 更新图像框的插槽
    def cv_update_image(self, cv_frame_img, cv_gray_img, cv_diff_img):
        # print('Update Img')
        qt_cv_frame_img = self.convert_bgr2qt(cv_frame_img)
        self.cv_frame_label.setPixmap(qt_cv_frame_img)

        qt_cv_gray_img = self.convert_cv_gray_qt(cv_gray_img)
        self.cv_gray_label.setPixmap(qt_cv_gray_img)

        qt_cv_diff_img = self.convert_cv_gray_qt(cv_diff_img)
        self.cv_diff_label.setPixmap(qt_cv_diff_img)
    
    @pyqtSlot(np.ndarray, int, int)
    def cv_update_piece(self, cv_piece_img, gray_scale, area_scale):
        qt_cv_piece_img = self.convert_cv_gray_qt(cv_piece_img)
        self.cv_piece_label.setPixmap(qt_cv_piece_img)
        self.cv_piece_gray_scale_label.setText(('Gray Scale: %d' % gray_scale))
        self.cv_piece_area_scale_label.setText(('Area Scale: %d' % area_scale))

    @pyqtSlot(np.ndarray)
    def yolo_update_image(self, yolo_update_img):
        qt_yolo_update_img = self.convert_bgr2qt(yolo_update_img)
        self.yolo_label.setPixmap(qt_yolo_update_img)

    @pyqtSlot(bool)
    def yolo_launched_check(self, bool):
        self.cv_thread.start()

    def convert_bgr2qt(self, cv_img):        # 将三通道 CV 图像转为 Qt 图像的方法
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def convert_cv_gray_qt(self, cv_img):       # 将单通道（灰色） CV 图像转为 Qt 图像的方法
        h, w = cv_img.shape
        bytes_per_line = w
        convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def buttonClicked(self):                    # 切换视频输入的方法
        global cv_selected_video_str
        sender = self.sender()

if __name__ == '__main__':
    main()