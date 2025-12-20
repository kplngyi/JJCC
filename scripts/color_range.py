import cv2
import numpy as np

def pick_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("图片路径无效")
        return

    # 转换为 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取点击点的 HSV 值
            hsv_value = hsv[y, x]
            # 获取点击点的 BGR 值（用于对比）
            bgr_value = img[y, x]
            print(f"坐标: ({x}, {y}) | BGR: {bgr_value} | 🎯 OpenCV-HSV: {hsv_value}")
            print(f"建议范围建议: [{max(0, hsv_value[0]-5)}, 100, 100] 到 [{min(179, hsv_value[0]+5)}, 255, 255]\n")

    cv2.namedWindow('Color Picker')
    cv2.setMouseCallback('Color Picker', mouse_callback)

    print("--- 提示: 点击图片中的黄色圆圈查看 HSV 值，按 'q' 退出 ---")
    while True:
        cv2.imshow('Color Picker', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# 替换为你本地保存这两张图片的路径
pick_color("image_19888c.png")