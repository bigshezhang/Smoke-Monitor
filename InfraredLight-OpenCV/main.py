#-*-coding:utf-8-*-

import sys
import _thread

from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

import select_interface
import user_interface


# class Main:
    # app = QApplication(sys.argv)
    # cd = video.ContourDetection()
    # def main(self):
    #     _thread.start_new_thread(self.cd.capture_contours, () )
    #     ui = user_interface.UserInterface()
    #     sys.exit(self.app.exec())

def main():
    app = QApplication(sys.argv)
    select_interface.SelectInterface()
    user_interface.ui = user_interface.UserInterface()
    user_interface.ui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()