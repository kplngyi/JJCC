import cv2
import os

def crop_templates_pinyin(image_path, output_dir="/Users/hpyi/Hobby/JJCC/assets/templates"):
    # 读取截图
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{image_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 核心坐标配置 (基于 JJ 象棋截图比例) ---
    start_x = 123   # 左上角第一个棋子的中心 X
    start_y = 488  # 左上角第一个棋子的中心 Y
    grid_w = 88   # 格子横向间距
    grid_h = 88   # 格子纵向间距
    size = 90      # 裁剪图片的大小 (90x90)

    # 棋子定义 (行索引 0-9, 列索引 0-8, 拼音文件名)
    # 红方在上方 (0-4行)，黑方在下方 (5-9行)
    pieces = [
        # --- 红方 (r_) ---
        (0, 0, "r_che"), (0, 1, "r_ma"), (0, 2, "r_xiang"), (0, 3, "r_shi"), (0, 4, "r_shuai"),
        (2, 1, "r_pao"), 
        (3, 0, "r_bing"),
        
        # --- 黑方 (b_) ---
        (9, 0, "b_che"), (9, 1, "b_ma"), (9, 2, "b_xiang"), (9, 3, "b_shi"), (9, 4, "b_jiang"),
        (7, 1, "b_pao"), 
        (6, 0, "b_zu")
    ]

    for r, c, name in pieces:
        # 计算该棋子的中心像素坐标
        center_y = start_y + r * grid_h
        center_x = start_x + c * grid_w
        
        # 计算裁剪边界
        y1, y2 = int(center_y - size//2), int(center_y + size//2)
        x1, x2 = int(center_x - size//2), int(center_x + size//2)
        
        # 执行裁剪
        crop = img[y1:y2, x1:x2]
        
        if crop.size > 0:
            save_path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(save_path, crop)
            print(f"成功保存拼音模板: {name}.png")
        else:
            print(f"错误: {name} 裁剪超出图片边界")

if __name__ == "__main__":
    # 使用你项目中的实际截图路径
    target_img = "/Users/hpyi/Hobby/JJCC/assets/screenshots/AIRWeChat_JJ象棋_2802.png"
    crop_templates_pinyin(target_img)