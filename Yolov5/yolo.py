import cv2
import numpy as np
import torch
import os

from .models.common import DetectMultiBackend
from .utils.general import check_img_size,non_max_suppression
from .utils.plots import Annotator, colors
from .utils.torch_utils import select_device
from .utils.augmentations import letterbox #调整图片大小至640
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from loguru import logger

class YoloDetection(QThread):
    logger.remove(handler_id=None)
    logger.add("yolo_info.log", format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{message}</level>', rotation = "50MB", enqueue = True)
    yolo_change_pixmap_signal = pyqtSignal(np.ndarray)     # 信号，用于提醒更新界面中的图片框

    #capture.set(cv2.CAP_PROP_BRIGHTNESS,50)#亮度
    #capture.set(cv2.CAP_PROP_CONTRAST,18)#对比度
    #capture.set(cv2.CAP_PROP_SATURATION,70)# 图像的饱和度（仅适用于相机）
    #capture.set(cv2.CAP_PROP_EXPOSURE,200)
    #可以去调调参数

    def run(self):
        print("已调用物品检测模块")
        # Load model
        device = select_device('')
        weights=os.getcwd() + "/Yolov5/weights/all_1.pt"
        dnn = False
        data=os.getcwd() + "/Yolov5/smoke/ps_1/config.ymal"
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        model.warmup()  # warmup
        capture = cv2.VideoCapture(os.getcwd() + '/Yolov5/videos/simulation.mp4')
        
        while (True):
        # 获取一帧
            ret, frame = capture.read()
            if ret == False:
                continue
            frame , ratio, (dw, dh)= letterbox(frame)

            img0=frame
            img = letterbox(frame)[0] #返回的是元组所以[0]


            frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            # Convert
            img = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img) #用tensor的说法 转为张量？

            im = torch.from_numpy(img).to(device)
            #im=im.half()
            im=im.float()  # uint8 to fp16/32 #转为float，除以255可以得小数
            im /= 255  # 0 - 255 to 0.0 - 1.0 #归一化

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

    		#im已经是预处理后的张量了，这才符合网络输入，而img0表示位图，帧
            pred = model(im, augment=False,visualize=False)#augmented inference  # visualize features增强推理#可视化特征
            # NMS 非极大值抑制
            pred = non_max_suppression(pred)
            det=pred[0]
            annotator = Annotator(frame, line_width=3, example=str(names))
            for *xyxy, conf, cls in iter(det):#一个图片里面可能不止一个目标对象，比如两个人，比如一人一狗，所以用循环
                c=int(cls)
                label =names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))
                logger.info("alert_x1 = {alert_x1}, alert_y1 = {alert_y1}, alert_x2 = {alert_x2}, alert_y2 = {alert_y2}" \
                      .format(alert_x1=round(xyxy[0].tolist() / frame_width, 3), alert_y1=round(xyxy[1].tolist() / frame_height, 3), \
                        alert_x2=round(xyxy[2].tolist() / frame_width, 3), alert_y2=round(xyxy[3].tolist() / frame_height, 3)))
            im0 = annotator.result()
            self.yolo_change_pixmap_signal.emit(im0)


    

