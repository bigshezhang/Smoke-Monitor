#!/bin/bash

uNames=`uname -s`
osName=${uNames: 0: 4}
if [ "$osName" == "Darw" ] # Darwin
then
    conda install grpcio pytorch torchvision torchaudio -c pytorch -y;
	pip3 install -r Yolov5/requirements.txt;
	python3 main.py
elif [ "$osName" == "Linu" ] # Linux
then
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
	pip3 install -r Yolov5/requirements.txt;
	python3 main.py
elif [ "$osName" == "MING" ] # MINGW, windows, git-bash
then 
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
	pip3 install -r Yolov5/requirements.txt;
	python3 main.py
else
	echo "unknown os"
fi

