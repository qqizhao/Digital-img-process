import cv2
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt, QUrl, QTimer

from window1 import Window1
from window2 import Window2
from window3 import Window3
from window4 import Window4

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('./ui/mainWindow.ui')
        self.ui.pushButton_1.clicked.connect(self.open_window1)
        self.ui.pushButton_2.clicked.connect(self.open_window2)
        self.ui.pushButton_3.clicked.connect(self.open_window3)
        self.ui.pushButton_4.clicked.connect(self.open_window4)
        
    def open_window1(self):
        self.window1 = Window1()
        self.window1.ui1.show()
        self.hide()
        
    def open_window2(self):
        self.window2 = Window2()
        self.window2.ui2.show()
        self.hide()
        
    def open_window3(self):
        self.window3 = Window3()
        self.window3.ui3.show()
        self.hide()
        
    def open_window4(self):
        self.window4 = Window4()
        self.window4.ui4.show()
        self.hide()

if __name__ == '__main__':
    app = QApplication([])
    main_window = MainWindow()
    main_window.ui.show()
    app.exec_()