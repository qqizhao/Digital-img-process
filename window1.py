import cv2
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import QSize
from PySide2.QtCore import Qt, QUrl, QTimer
import numpy as np

class Window1(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui1 = QUiLoader().load('./ui/window1.ui')
        # self.ui1.pushButton_2.clicked.connect(self.open_window2)
        self.ui1.select.clicked.connect(self.select_handler)
        self.ui1.show1.clicked.connect(self.show_handler)
        self.ui1.save.clicked.connect(self.save_handler)

    def select_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.input_filename, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                             "All Files (*.png *.jpg *.bmp *.mp4)",
                                                             options=options)
        if self.input_filename:
            if self.input_filename.endswith(('.png', '.jpg', '.bmp')):
                pixmap = QPixmap(self.input_filename)
                label_size = self.ui1.input_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui1.input_label.setPixmap(scaled_pixmap)
                
                
    def show_handler(self):
        if self.ui1.is_gray.isChecked():
            img = cv2.imread(self.input_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_qpixmap = QImage(img_gray.data, img_gray.shape[1], img_gray.shape[0], QImage.Format_Grayscale8)
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui1.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui1.output_label.setPixmap(scaled_pixmap)
            print("Gray")
            
        elif self.ui1.is_binary.isChecked():
            img = cv2.imread(self.input_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            img_qpixmap = QImage(img_bin.data, img_bin.shape[1], img_bin.shape[0], QImage.Format_Grayscale8)
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui1.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui1.output_label.setPixmap(scaled_pixmap)
            print("Binary")
            
        elif self.ui1.is_trans.isChecked():
            img = cv2.imread(self.input_filename)
            choice = self.ui1.transform.currentText()
            label_size = self.ui1.output_label.size()
            width, height, channels = img.shape[:]
            
            if choice == '缩小':
                img_s = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                new_height, new_width, _ = img_s.shape
                x_offset = int((img.shape[1] - new_width) / 2)
                y_offset = int((img.shape[0] - new_height) / 2)
                img_qpixmap = np.zeros((height, width, channels), dtype=np.uint8)
                img_qpixmap[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_s 
                print('缩小')              
            elif choice == '放大':
                img_b = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                new_height, new_width, _ = img_b.shape
                x_start = int((new_width - img.shape[1]) / 2)
                x_end = x_start + img.shape[1]
                y_start = int((new_height - img.shape[0]) / 2)
                y_end = y_start + img.shape[0]
                img_qpixmap = img_b[y_start:y_end, x_start:x_end]
                cv2.imwrite('./data/lena_big.jpg', img_qpixmap)
                print('放大')
            elif choice == '旋转':
                rows, cols = img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
                img_qpixmap = cv2.warpAffine(img, M, (cols, rows))
                print('旋转')
            else:
                rows, cols = img.shape[:2]
                M = np.float32([[1, 0, 100], [0, 1, 50]])
                img_qpixmap = cv2.warpAffine(img, M, (cols, rows))
                print('平移')
                           
            if choice == '放大':
                img_qpixmap = QImage('./data/lena_big.jpg')    
            else:
                height, width, channels = img_qpixmap.shape
                img_qpixmap = QImage(img_qpixmap.data, width, height, width * channels, QImage.Format_BGR888)
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui1.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui1.output_label.setPixmap(scaled_pixmap)
            
    def save_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        output_filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JPEG Files (*.jpg);;PNG Files (*.png)", options=options)

        if output_filename:
            self.pixmap.save(output_filename)