import cv2
import os
from collections import defaultdict

# 棋盘：10 行 × 9 列
# None 表示该位置没棋子
board = [
    ["r_che","r_ma","r_xiang","r_shi","r_shuai","r_shi","r_xiang","r_ma","r_che"],
    [None,None,None,None,None,None,None,None,None],
    [None,"r_pao",None,None,None,None,None,"r_pao",None],
    ["r_bing",None,"r_bing",None,"r_bing",None,"r_bing",None,"r_bing"],
    [None]*9,
    [None]*9,
    ["b_zu",None,"b_zu",None,"b_zu",None,"b_zu",None,"b_zu"],
    [None,"b_pao",None,None,None,None,None,"b_pao",None],
    [None,None,None,None,None,None,None,None,None],
    ["b_che","b_ma","b_xiang","b_shi","b_jiang","b_shi","b_xiang","b_ma","b_che"],
]

def crop_templates_pinyin(image_path, output_dir="/Users/hpyi/Hobby/JJCC/assets/templates"):

    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return

    os.makedirs(output_dir, exist_ok=True)

    start_x = 123
    start_y = 488
    grid_w = 90
    grid_h = 87.2
    size = 90

    counter = defaultdict(int)

    for r in range(10):
        for c in range(9):
            name = board[r][c]
            if name is None:
                continue

            center_y = start_y + r * grid_h
            center_x = start_x + c * grid_w

            y1, y2 = int(center_y - size//2), int(center_y + size//2)
            x1, x2 = int(center_x - size//2), int(center_x + size//2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            idx = counter[name]
            counter[name] += 1

            save_path = os.path.join(output_dir, f"{name}_{idx}.png")
            cv2.imwrite(save_path, crop)
            print(f"保存: {name}_{idx}.png")

if __name__ == "__main__":
    # 使用你项目中的实际截图路径
    target_img = "/Users/hpyi/Hobby/JJCC/assets/screenshots/AIRWeChat_JJ象棋_2802.png"
    crop_templates_pinyin(target_img)