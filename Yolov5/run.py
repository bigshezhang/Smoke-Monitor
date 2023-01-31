import cv2
import numpy as np
import torch
import os
from models.common import DetectMultiBackend
from utils.general import check_img_size,non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox #调整图片大小至640

if __name__ == '__main__':
    # Load model
    device = select_device('')
    weights="weights/all_1.pt"
    dnn = False
    data="smoke/ps_1/config.ymal"
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    model.warmup()  # warmup

    capture = cv2.VideoCapture(os.getcwd() + '/videos/simulation.mp4')
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 计算视频的高  # 获取视频宽度
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 计算视频的宽  # 获取视频高度

 	#https://blog.csdn.net/weixin_41010198/article/details/88535234
    #capture.set(cv2.CAP_PROP_BRIGHTNESS,50)#亮度
    #capture.set(cv2.CAP_PROP_CONTRAST,18)#对比度
    #capture.set(cv2.CAP_PROP_SATURATION,70)# 图像的饱和度（仅适用于相机）
    #capture.set(cv2.CAP_PROP_EXPOSURE,200)
    #如果你感兴趣可以去调调参数


    while (True):
        # 获取一帧
        ret, frame = capture.read()
        frame , ratio, (dw, dh)= letterbox(frame)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将这帧转换为灰度图
        frame = cv2.flip(frame, 1)   #cv2.flip 图像翻转

        img0=frame
        img = letterbox(frame)[0] #返回的是元组所以[0]
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
        im0 = annotator.result()
        cv2.imshow('frame',im0)

        # 如果输入q，则退出
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

