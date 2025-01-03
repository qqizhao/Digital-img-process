import cv2
import matplotlib.pyplot as plt
import numpy as np

# 显示直方图
img = cv2.imread('./data/lena.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.savefig('./data/exp2/hist.png')
plt.show()

# 直方图均衡化
img_equ = cv2.equalizeHist(img_gray)

cv2.imshow('img_equalize',img_equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist_equ = cv2.calcHist([img_equ], [0], None, [256], [0, 256])


plt.plot(hist_equ)
plt.savefig('./data/exp2/hist_equ.png')
plt.show()

# 利用模板进行空域滤波
# 1.平滑滤波
## 均值滤波
img_blur = cv2.blur(img,(3,5))
cv2.imshow('img_blur',img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_box = cv2.boxFilter(img, -1, (3,5))
cv2.imshow('img_box',img_box)
cv2.waitKey(0)
cv2.destroyAllWindows()

### 高斯模糊滤波
img_gauss = cv2.GaussianBlur(img,(3,5),0)
cv2.imshow('img_gauss',img_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

### 中值滤波
img_median = cv2.medianBlur(img,5)
cv2.imshow('img_median',img_median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.锐化滤波
## Roberts算子
ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
kernelx = np.array([[-1, 0], [0, 1]])
kernely = np.array([[0, -1], [1, 0]])
x_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernelx)
y_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernely)
# 转uint8
absX_Robert = cv2.convertScaleAbs(x_Robert)
absY_Robert = cv2.convertScaleAbs(y_Robert)
img_Robert = cv2.addWeighted(absX_Robert, 0.5, absY_Robert, 0.5, 0)
cv2.imwrite('./data/exp2/img_Robert.png', img_Robert)
cv2.imshow('img_Robert',img_Robert)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Prewitt算子
kernelx_Prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely_Prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x_Prewitt = cv2.filter2D(img_binary, cv2.CV_16S, kernelx_Prewitt)
y_Prewitt = cv2.filter2D(img_binary, cv2.CV_16S, kernely_Prewitt)
absX_Prewitt = cv2.convertScaleAbs(x_Prewitt)
absY_Prewitt = cv2.convertScaleAbs(y_Prewitt)
img_Prewitt = cv2.addWeighted(absX_Prewitt, 0.5, absY_Prewitt, 0.5, 0)
cv2.imwrite('./data/exp2/img_Prewitt.png', img_Prewitt)
cv2.imshow("img_Prewitt", img_Prewitt)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Sobel算子
x_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 1, 0)
y_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 0, 1)
absX_Sobel = cv2.convertScaleAbs(x_Sobel)
absY_Sobel = cv2.convertScaleAbs(y_Sobel)
img_Sobel = cv2.addWeighted(absX_Sobel, 0.5, absY_Sobel,0.5, 0)
cv2.imwrite('./data/exp2/img_Sobel.png', img_Sobel)
cv2.imshow("img_Sobel", img_Sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()


## 3.滤波器滤波
### 低通滤波器
def lowPassFilter(img,size):
    h,w = img.shape[:2]
    h_center, w_center = int(h/2), int(w/2)
    img_black = np.zeros((h,w),dtype=np.uint8)
    img_black[h_center-int(size/2):h_center+int(size/2), w_center-int(size/2):w_center+int(size/2)] = 1
    out_img = img * img_black
    return out_img

def highPassFilter(img,size):
    h,w = img.shape[:2]
    h_center, w_center = int(h/2), int(w/2)
    output_img = img
    output_img[h_center-int(size/2):h_center+int(size/2), w_center-int(size/2):w_center+int(size/2)] = 0
    return output_img

img_dft = np.fft.fft2(img_gray)
dft_shift = np.fft.fftshift(img_dft)
dft_shift_low = lowPassFilter(dft_shift, 100)
res = np.log(np.abs(dft_shift_low))
idft_shift= np.fft.ifftshift(dft_shift_low)
img_low = np.fft.ifft2(idft_shift)
img_low = np.abs(img_low)
cv2.imwrite('./data/exp2/img_lowPassFilter.png', img_low)
cv2.imshow("img_lowPassFilter", np.int8(img_low))
cv2.waitKey(0)
cv2.destroyAllWindows()

dft_shift_high = highPassFilter(dft_shift, 50)
res = np.log(np.abs(dft_shift_high))
idft_shift= np.fft.ifftshift(dft_shift_high)
img_high = np.fft.ifft2(idft_shift)
img_high = np.abs(img_high)
cv2.imwrite('./data/exp2/img_highPassFilter.png', img_high)
cv2.imshow("img_highPassFilter", np.int8(img_high))
cv2.waitKey(0)
cv2.destroyAllWindows()


    