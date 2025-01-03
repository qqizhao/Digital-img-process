import cv2

img = cv2.imread('./data/exp4.png')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# LOG 检测器
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(3,3),1,1)

log_result = cv2.Laplacian(img_blur, cv2.CV_16S, ksize=1)
img_log = cv2.convertScaleAbs(log_result)
cv2.imwrite('./data/exp4/img_log.png', img_log)
cv2.imshow('image_log', img_log)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Scharr
Scharr_result = cv2.Scharr(img_gray, cv2.CV_16S, 1, 0)
img_scharr = cv2.convertScaleAbs(Scharr_result)
cv2.imwrite('./data/exp4/img_scharr.png', img_scharr)
cv2.imshow('image_Scharr', img_scharr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny
img_blur_canny = cv2.GaussianBlur(img_gray,(7,7),1,1)
img_Canny = cv2.Canny(img_blur_canny, 50, 150)
cv2.imwrite('./data/exp4/img_Canny.png', img_Canny)
cv2.imshow('image_canny', img_Canny)
cv2.waitKey(0)
cv2.destroyAllWindows()