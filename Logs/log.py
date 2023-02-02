#!/usr/bin/env python

import pandas as pd
import os
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import schedule

class LogComparison(QThread):
    log_alert_signal = pyqtSignal(bool, list) 
    def run(self):
        print("日志比较线程已启动")
        schedule.every(1).seconds.do(self.clean_up_logs)    # 每 1 秒清理一次日志，保证读入的数据均为最近一秒的
        while(True):
            schedule.run_pending()  # 启动定时任务
            try:
                cv=pd.read_csv('Logs/cv_info.log',header=None)  # 读取日志软件，cv_info 对应的 cv_storage 是持久化的数据
                yolo=pd.read_csv('Logs/yolo_info.log',header=None)
                res=[]
            except:
                time.sleep(0.001)
                continue
            cv['time']=cv.iloc[:,0].apply(lambda x : x.split('|')[0].split('.')[0].strip()).astype('datetime64[ns]')#pd.to_datetime(,format='%Y-%m-%d %H:%M:%s')
            cv['cv_x1']=cv.iloc[:,0].apply(lambda x : x.split('=')[-1].strip())
            cv['cv_y1']=cv.iloc[:,1].apply(lambda x : x.split('=')[-1].strip())
            cv['cv_x2']=cv.iloc[:,2].apply(lambda x : x.split('=')[-1].strip())
            cv['cv_y2']=cv.iloc[:,3].apply(lambda x : x.split('=')[-1].strip())
            cv_single=cv.drop_duplicates(['time'],keep ='first')
            cv_single=cv_single[['time','cv_x1','cv_y1','cv_x2','cv_y2']].reset_index(drop=True)

            yolo['time']=yolo.iloc[:,0].apply(lambda x : x.split('|')[0].split('.')[0].strip()).astype('datetime64[ns]')#pd.to_datetime(,format='%Y-%m-%d %H:%M:%s')
            yolo['yolo_x1']=yolo.iloc[:,0].apply(lambda x : x.split('=')[-1].strip())
            yolo['yolo_y1']=yolo.iloc[:,1].apply(lambda x : x.split('=')[-1].strip())
            yolo['yolo_x2']=yolo.iloc[:,2].apply(lambda x : x.split('=')[-1].strip())
            yolo['yolo_y2']=yolo.iloc[:,3].apply(lambda x : x.split('=')[-1].strip())
            yolo_single=yolo.drop_duplicates(['time'],keep ='first')
            yolo_single=yolo_single[['time','yolo_x1','yolo_y1','yolo_x2','yolo_y2']].reset_index(drop=True)

            yolo_s=yolo_single.set_index('time')
            cv_s=cv_single.set_index('time')
            data=pd.concat([yolo_s,cv_s],axis=1,join='inner')
            for i in range(data.shape[0]):
                tem=self.iou(data.iloc[i,0:4].tolist(),data.iloc[i,4:].tolist())
                # print(data.iloc[i,0:4].tolist())
                # print(data.iloc[i,4:].tolist())
                if tem[0]:
                    self.log_alert_signal.emit(True, tem[1])
                else:
                    self.log_alert_signal.emit(False, [])
            time.sleep(0.5)


    def iou(self, R1, R2):
        w1=float(R1[2])-float(R1[0])
        h1=float(R1[3])-float(R1[1])
        w2=float(R2[2])-float(R2[0])
        h2=float(R2[3])-float(R2[1])
        w=abs((float(R1[0])+float(R1[2])) /2 - (float(R2[0])+float(R2[2])) / 2)
        h=abs((float(R1[1])+float(R1[3])) /2 - (float(R2[1])+float(R2[3])) / 2)
        x1=float(max(R1[0],R2[0]))
        y1=float(max(R1[1],R2[1]))
        x2=float(min(R1[2],R2[2]))
        y2=float(min(R1[3],R2[3]))
        if w < (w1+w2) / 2 and h < (h1+h2) / 2:
            # print([x1, y1, x2, y2])
            return True,[x1, y1, x2, y2]
        else:
            return False,[]

    def clean_up_logs(self):    # 清理日志
        yolo_log = open('Logs/yolo_info.log', "r+")
        yolo_log.truncate()

        cv_log = open('Logs/cv_info.log', "r+")
        cv_log.truncate()
        # print("已清理日志")

