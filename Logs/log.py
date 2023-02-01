#!/usr/bin/env python

import pandas as pd
import os
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import schedule

class LogComparison(QThread):
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
                if self.solve(data.iloc[i,0:4].tolist(),data.iloc[i,4:].tolist()):
                    res.append(str(data.index[i])+'有重合❌')
                    print(time.time(), " [有重合]")
                else:
                    res.append(str(data.index[i])+'无重合✅')
                    print(time.time(), " [无重合]")

            time.sleep(0.5)


    def solve(self, R1, R2):
        if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
            return False
        else:
            return True

    def clean_up_logs(self):    # 清理日志
        yolo_log = open('Logs/yolo_info.log', "r+")
        yolo_log.truncate()

        cv_log = open('Logs/cv_info.log', "r+")
        cv_log.truncate()
        print("已清理日志")

