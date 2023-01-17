#-*-coding:utf-8-*-
import video
import select_interface

def main():
    select_interface.SelectInterface().hello_interface()
    video.ContourDetection().capture_contours()

if __name__ == '__main__':
    main()