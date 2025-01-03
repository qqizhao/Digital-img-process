import cv2
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt, QUrl, QTimer

import numpy as np
import matplotlib.pyplot as plt

class Window3(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui3 = QUiLoader().load('./ui/window3.ui')
        self.ui3.select.clicked.connect(self.select_handler)
        self.ui3.show1.clicked.connect(self.show_handler)
        self.ui3.save.clicked.connect(self.save_handler)
        self.ui3.show_all.clicked.connect(self.show_all_handler)
    
    def show_all_handler(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        img = cv2.imread('./data/exp3/img_all_exp3.png')
        cv2.resizeWindow('image', int(img.shape[1]/2), int(img.shape[0]/2))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    
    def select_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.input_filename, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                             "All Files (*.png *.jpg *.bmp *.mp4)",
                                                             options=options)
        if self.input_filename:
            if self.input_filename.endswith(('.png', '.jpg', '.bmp')):
                pixmap = QPixmap(self.input_filename)
                label_size = self.ui3.input_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui3.input_label.setPixmap(scaled_pixmap)
    
    def show_handler(self):
        img = cv2.imread(self.input_filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)

        if self.ui3.is_erode.isChecked():
            img_qpixmap = cv2.erode(img_binary, kernel, iterations=1)
            cv2.imwrite('./data/exp3/erode.png', img_qpixmap)
            self.output_filename = './data/exp3/erode.png'
            print('Erode')
        
        elif self.ui3.is_dilate.isChecked():
            img_qpixmap = cv2.dilate(img_binary, kernel, iterations=1)
            cv2.imwrite('./data/exp3/dilate.png', img_qpixmap)
            self.output_filename = './data/exp3/dilate.png'
            print('Dilate')
            
        elif self.ui3.is_open.isChecked():
            img_qpixmap = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
            cv2.imwrite('./data/exp3/open.png', img_qpixmap)
            self.output_filename = './data/exp3/open.png'
            print('Open')
        
        else:
            img_qpixmap = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite('./data/exp3/close.png', img_qpixmap)
            self.output_filename = './data/exp3/close.png'
            print('Close')
            
        img_qpixmap = QImage(self.output_filename)    
        self.pixmap = QPixmap(img_qpixmap)
        label_size = self.ui3.output_label.size()
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui3.output_label.setPixmap(scaled_pixmap)  
    
    def save_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        output_filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JPEG Files (*.jpg);;PNG Files (*.png)", options=options)

        if output_filename:
            self.pixmap.save(output_filename)