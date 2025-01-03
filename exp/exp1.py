import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取并显示图像
img = cv2.imread('./data/lena.png')

print('Pic_size:',img.shape)
print('Pic_type:',img.dtype)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 对灰度图进行二值化处理
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('binary image', img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 对图像进行几何变换
# 放大或缩小
scale_x = 2
scale_y = 2
img_2x = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
print('Pic_size:',img_2x.shape)
cv2.imwrite('./data/exp1/lena_2x.png', img_2x)

# 旋转
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
rotated_img = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('./data/exp1/lena_rotate.png', img_2x)
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 平移
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated_img = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('./data/exp1/lena_trans.png', img_2x)
cv2.imshow('Translated Image', translated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()