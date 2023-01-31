#-*-coding:utf-8-*-

import sys
from PyQt6.QtWidgets import QApplication

import user_interface

def main():
    app = QApplication(sys.argv)
    user_interface.ui = user_interface.UserInterface()
    user_interface.ui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()