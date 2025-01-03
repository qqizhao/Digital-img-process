import cv2
from PySide2 import QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage, QIcon
from PySide2.QtCore import Qt, QUrl, QTimer

import numpy as np
import matplotlib.pyplot as plt

class Window2(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui2 = QUiLoader().load('./ui/window2.ui')
        # self.ui2.pushButton_2.clicked.connect(self.open_window2)
        self.ui2.select.clicked.connect(self.select_handler)
        self.ui2.show1.clicked.connect(self.show_handler)
        self.ui2.save.clicked.connect(self.save_handler)
        self.ui2.show_all.clicked.connect(self.show_all_handler)
    
    def show_all_handler(self):
        if self.ui2.is_pinghua.isChecked():
            img = cv2.imread('./data/exp2/img_all_exp2_pinghua.png')
            
        elif self.ui2.is_ruihua.isChecked():
            img = cv2.imread('./data/exp2/img_all_exp2_suanzi.png')
            
        elif self.ui2.is_filter.isChecked():
            img = cv2.imread('./data/exp2/img_all_exp2_filter.png')
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
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
                label_size = self.ui2.input_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui2.input_label.setPixmap(scaled_pixmap)
                
    def show_handler(self):
        if self.ui2.is_hist.isChecked():
            img = cv2.imread(self.input_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            choice = self.ui2.equhist.currentText()
            
            if choice == '直方图':
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                plt.clf()  # 清空图像窗口
                plt.plot(hist)
                plt.savefig('./data/hist1.png')
            else:
                img_equ = cv2.equalizeHist(img_gray)
                img_equ_hist = cv2.calcHist([img_equ], [0], None, [256], [0, 256])
                plt.clf()  # 清空图像窗口
                plt.plot(img_equ_hist)
                plt.savefig('./data/hist2.png')
                
            if choice == '直方图':
                img_qpixmap = QImage('./data/hist1.png')    
            else:
                img_qpixmap = QImage('./data/hist2.png')  
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui2.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui2.output_label.setPixmap(scaled_pixmap)  
        
        elif self.ui2.is_pinghua.isChecked():
            img = cv2.imread(self.input_filename)
            choice = self.ui2.pinghua.currentText()
            
            if choice == '均值滤波':
                # img_blur = cv2.blur(img,(3,5))
                img_qpixmap = cv2.boxFilter(img, -1, (3,5))
                print('均值滤波')

            elif choice == '高斯模糊滤波':     
                img_qpixmap = cv2.GaussianBlur(img,(3,5),0)
                print('高斯模糊滤波')
               
            else:
                img_qpixmap = cv2.medianBlur(img,5)
                print('中值滤波')
            
            height, width, channels = img_qpixmap.shape
            img_qpixmap = QImage(img_qpixmap.data, width, height, width * channels, QImage.Format_BGR888)    
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui2.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui2.output_label.setPixmap(scaled_pixmap)  
            
        elif self.ui2.is_ruihua.isChecked():
            img = cv2.imread(self.input_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            choice = self.ui2.ruihua.currentText()        
            
            if choice == 'Roberts算子':
                
                kernelx = np.array([[-1, 0], [0, 1]])
                kernely = np.array([[0, -1], [1, 0]])
                x_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernelx)
                y_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernely)
                absX_Robert = cv2.convertScaleAbs(x_Robert)
                absY_Robert = cv2.convertScaleAbs(y_Robert)
                img_qpixmap = cv2.addWeighted(absX_Robert, 0.5, absY_Robert, 0.5, 0)
                print('Roberts算子')
                
            elif choice == 'Prewitt算子':
                kernelx_Prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
                kernely_Prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
                x_Prewitt = cv2.filter2D(img_binary, cv2.CV_16S, kernelx_Prewitt)
                y_Prewitt = cv2.filter2D(img_binary, cv2.CV_16S, kernely_Prewitt)
                absX_Prewitt = cv2.convertScaleAbs(x_Prewitt)
                absY_Prewitt = cv2.convertScaleAbs(y_Prewitt)
                img_qpixmap = cv2.addWeighted(absX_Prewitt, 0.5, absY_Prewitt, 0.5, 0)
                print('Prewitt算子')
            
            else:
                x_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 1, 0)
                y_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 0, 1)
                absX_Sobel = cv2.convertScaleAbs(x_Sobel)
                absY_Sobel = cv2.convertScaleAbs(y_Sobel)
                img_qpixmap = cv2.addWeighted(absX_Sobel, 0.5, absY_Sobel,0.5, 0)
                print('Sobel算子')
                
                
            img_qpixmap = QImage(img_qpixmap.data, img_qpixmap.shape[1], img_qpixmap.shape[0], QImage.Format_Grayscale8)
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui2.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui2.output_label.setPixmap(scaled_pixmap)
        
        else:
            img = cv2.imread(self.input_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            choice = self.ui2.filter.currentText()     
            img_dft = np.fft.fft2(img_gray)
            dft_shift = np.fft.fftshift(img_dft)   
            
            if choice == '低通滤波器':
                dft_shift_low = self.lowPassFilter(dft_shift, 100)
                # res = np.log(np.abs(dft_shift_low))
                idft_shift= np.fft.ifftshift(dft_shift_low)
                img_low = np.fft.ifft2(idft_shift)
                img_qpixmap = np.abs(img_low)
                cv2.imwrite('./data/filter.jpg', img_qpixmap)
                print('低通滤波器')
        
            else:
                dft_shift_high = self.highPassFilter(dft_shift, 50)
                # res = np.log(np.abs(dft_shift_high))
                idft_shift= np.fft.ifftshift(dft_shift_high)
                img_high = np.fft.ifft2(idft_shift)
                img_qpixmap = np.abs(img_high)
                cv2.imwrite('./data/filter.jpg', img_qpixmap)
                print('高通滤波器')
                
            img_qpixmap = QImage('./data/filter.jpg')    
            self.pixmap = QPixmap(img_qpixmap)
            label_size = self.ui2.output_label.size()
            scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui2.output_label.setPixmap(scaled_pixmap)  
       
            
    def save_handler(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        output_filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JPEG Files (*.jpg);;PNG Files (*.png)", options=options)

        if output_filename:
            self.pixmap.save(output_filename)
            
    def lowPassFilter(self,img,size):
                    h,w = img.shape[:2]
                    h_center, w_center = int(h/2), int(w/2)
                    img_black = np.zeros((h,w),dtype=np.uint8)
                    img_black[h_center-int(size/2):h_center+int(size/2), w_center-int(size/2):w_center+int(size/2)] = 1
                    out_img = img * img_black
                    return out_img

    def highPassFilter(self,img,size):
        h,w = img.shape[:2]
        h_center, w_center = int(h/2), int(w/2)
        output_img = img
        output_img[h_center-int(size/2):h_center+int(size/2), w_center-int(size/2):w_center+int(size/2)] = 0
        return output_img