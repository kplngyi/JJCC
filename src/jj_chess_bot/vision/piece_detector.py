import cv2
import numpy as np
import os
class PieceDetector:
    def __init__(self, template_dir_name="assets/templates"):
            # 1. 获取当前文件 (piece_detector.py) 的绝对路径
            current_file_path = os.path.abspath(__file__)
            
            # 2. 向上追溯到项目根目录 (JJCC)
            # piece_detector.py -> vision -> jj_chess_bot -> src -> JJCC (4层)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
            
            # 3. 拼接得到模板目录的绝对路径
            self.template_dir = os.path.join(root_dir, template_dir_name)
            
            self.templates = {}
            self._load_templates()
            print(f"Loaded {len(self.templates)} piece templates from {self.template_dir}")

    def _load_templates(self):
            if not os.path.exists(self.template_dir):
                print(f"错误: 找不到模板目录 {self.template_dir}")
                return

            for file in os.listdir(self.template_dir):
                if file.endswith(".png"):
                    name = file.split(".")[0]
                    path = os.path.join(self.template_dir, file)
                    # 使用 cv2 读取
                    img = cv2.imread(path)
                    if img is not None:
                        self.templates[name] = img
                        print(f"已加载模板: {name}")

    def find_pieces(self, screen_img_path, threshold=0.8):
        """
        在截图中寻找匹配的棋子
        """
        # 读取截图
        screen = cv2.imread(screen_img_path)
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        results = []

        for name, temp in self.templates.items():
            # 模板也转为灰度
            temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            w, h = temp_gray.shape[::-1]

            # 执行模板匹配
            res = cv2.matchTemplate(screen_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                # 记录找到的棋子名称和坐标 (x, y, w, h)
                results.append({
                    "name": name,
                    "pos": (pt[0], pt[1], w, h),
                    "confidence": res[pt[1], pt[0]]
                })
        
        return results