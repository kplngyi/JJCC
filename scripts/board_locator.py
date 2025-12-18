import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取截图
target_img = "/Users/hpyi/Hobby/JJCC/assets/screenshots/WeChat_JJ象棋_2802.png"
img = cv2.imread(target_img)

# 转为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# 膨胀/腐蚀，让边界闭合
kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大轮廓
max_contour = max(contours, key=cv2.contourArea)

# 逼近矩形
epsilon = 0.02 * cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, epsilon, True)

# 如果是矩形，直接切图
x, y, w, h = cv2.boundingRect(approx)
chessboard = img[y:y+h, x:x+w]

# 可选：显示结果
plt.imshow(cv2.cvtColor(chessboard, cv2.COLOR_BGR2RGB))
plt.show()

# 保存切图
cv2.imwrite("chessboard.png", chessboard)