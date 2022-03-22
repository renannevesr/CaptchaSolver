import cv2
import matplotlib.pylab as plt

img1 = cv2.imread("creta1.png")

# make image gray
gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

# imgCanny = cv2.Canny(imgBlur,10,70) # Aplica canny.

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 1)
bilateral = cv2.bilateralFilter(gray, 5, 75, 75)

# Thresholding
ret, thresh = cv2.threshold(
    bilateral, 25, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Kernal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cv2.imshow('Result', kernel)
cv2.waitKey(5000)
# other things
erosion = cv2.erode(thresh, kernel, iterations=1)
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=1)

# Transform image
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(
    dist_transform, 0.02*dist_transform.max(), 255, cv2.THRESH_BINARY)  # ,255,0)

# kernel_1
kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))

dilation_1 = cv2.dilate(sure_fg, kernel_1, iterations=2)
erosion_1 = cv2.erode(dilation_1, kernel_1, iterations=3)

plt.imshow(erosion_1, 'gray')

cv2.imshow('Result', erosion_1)
cv2.waitKey(5000)
cv2.imwrite("./output.jpg", erosion_1)
