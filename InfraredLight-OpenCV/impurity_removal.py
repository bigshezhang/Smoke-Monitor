#-*-coding:utf-8-*-

import cv2
import colorsys
import time
from PIL import Image 


class ImpurityRemoval:
    def interference_detection(self, frame_lwpCV) -> bool:     # 返回True代表为干扰，返回False代表不为干扰
        im = Image.fromarray(frame_lwpCV)
        im = im.convert('RGBA')
        im.show("233")
        dominant_color = None
        for count, (r, g, b, a) in im.getcolors(im.size[0] * im.size[1]):
        # 跳过纯黑色
            if a == 0:
                continue
         
            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        
            y = (y - 16.0) / (235 - 16)
         
            # 忽略高亮色
            if y > 0.9:
                continue
         
            # Calculate the score, preferring highly saturated colors.
            # Add 0.1 to the saturation so we don't completely ignore grayscale
            # colors by multiplying the count by zero, but still give them a low
            # weight.
            score = (saturation + 0.1) * count
         
        dominant_color = (r, g, b)
        print("Blue_Score_is " , b , "\n")
        cv2.imshow("Blue_Score_is %d" % (b), frame_lwpCV)
        time.sleep(3)
        return True
