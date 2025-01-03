import cv2
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt, QUrl, QTimer

import numpy as np
import matplotlib.pyplot as plt

class Window4(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui4 = QUiLoader().load('./ui/window4.ui')
        self.ui4.select.clicked.connect(self.select_handler)
        self.ui4.show1.clicked.connect(self.show_handler)
        self.ui4.save.clicked.connect(self.save_handler)
        self.ui4.show_all.clicked.connect(self.show_all_handler)
    
    def show_all_handler(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        img = cv2.imread('./data/exp4/img_all_exp4.png')
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
                label_size = self.ui4.input_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui4.input_label.setPixmap(scaled_pixmap)
    
    def show_handler(self):
        img = cv2.imread(self.input_filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       

        if self.ui4.is_log.isChecked():
            img_blur = cv2.GaussianBlur(img_gray,(3,3),1,1)
            log_result = cv2.Laplacian(img_blur, cv2.CV_16S, ksize=1)
            img_log = cv2.convertScaleAbs(log_result)
            cv2.imwrite('./data/exp4/log.png', img_log)
            self.output_filename = './data/exp4/log.png'
            print('LOG')
        
        elif self.ui4.is_scharr.isChecked():
            Scharr_result = cv2.Scharr(img_gray, cv2.CV_16S, 1, 0)
            img_scharr = cv2.convertScaleAbs(Scharr_result)
            cv2.imwrite('./data/exp4/img_scharr.png', img_scharr)
            self.output_filename = './data/exp4/img_scharr.png'
            print('Scharr')
            
        else:
            img_blur_canny = cv2.GaussianBlur(img_gray,(7,7),1,1)
            img_Canny = cv2.Canny(img_blur_canny, 50, 150)
            cv2.imwrite('./data/exp4/img_Canny.png', img_Canny)
            self.output_filename = './data/exp4/img_Canny.png'
            print('Canny')
   
        img_qpixmap = QImage(self.output_filename)    
        self.pixmap = QPixmap(img_qpixmap)
        label_size = self.ui4.output_label.size()
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui4.output_label.setPixmap(scaled_pixmap)  
    
    def save_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        output_filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JPEG Files (*.jpg);;PNG Files (*.png)", options=options)

        if output_filename:
            self.pixmap.save(output_filename)