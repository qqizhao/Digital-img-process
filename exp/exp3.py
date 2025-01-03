import cv2
import numpy as np

img = cv2.imread('./data/exp3.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
cv2.imwrite('./data/exp3_binary.png', img_binary)
cv2.imshow('img_binary', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows() 

kernel = np.ones((3, 3), np.uint8)

# 腐蚀操作
img_erosion = cv2.erode(img_binary, kernel, iterations=1)
cv2.imwrite('./data/exp3/img_erosion.png', img_erosion)
cv2.imshow('img_erosion', img_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 膨胀
img_dilation = cv2.dilate(img_binary, kernel, iterations=1)
cv2.imwrite('./data/exp3/img_dilation.png', img_dilation)
cv2.imshow('img_dilation', img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 开运算：先腐蚀后膨胀
img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite('./data/exp3/img_opening.png', img_opening)
cv2.imshow('img_opening', img_opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 闭运算：先膨胀后腐蚀
img_closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
print(img_closing.shape)
cv2.imshow('img_closed', img_closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('Multiple Images', cv2.WINDOW_NORMAL)
                                                                                                                                                      
# 设置窗口大小为负值，使其自适应调节到适合屏幕的大小
img_all = np.hstack((img_binary, img_erosion, img_dilation, img_opening, img_closing))
cv2.resizeWindow('Multiple Images', int(img_all.shape[1]/2), int(img_all.shape[0]/2))
cv2.imshow('Multiple Images', img_all)
cv2.waitKey(0)
cv2.destroyAllWindows()