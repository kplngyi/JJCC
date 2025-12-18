# from jj_chess_bot.vision.piece_detector import PieceDetector
# detector = PieceDetector()
# results = detector.find_pieces("test_screen.png", threshold=0.8)
# for res in results:
#     print(f"Found {res['name']} at {res['pos']} with confidence {res['confidence']:.2f}")
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class PieceDetector:
    def __init__(self, template_dir_name="assets/templates",model_path="/Users/hpyi/Hobby/JJCC/assets/chess_model.pth", device=None):
        # 1. 动态获取项目根目录下的模板路径
        current_file_path = os.path.abspath(__file__)
        # 追溯路径: piece_detector.py -> vision -> jj_chess_bot -> src -> JJCC
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        self.template_dir = os.path.join(root_dir, template_dir_name)
        # self.model_path = os.path.join(root_dir, model_path)
        print(f"模板目录路径: {self.template_dir}")
        # 2. 棋盘网格参数 (基于你的截图比例)
        self.grid_params = {
            "start_x": 123,   # 左上角第一个格点中心 X
            "start_y": 488,  # 左上角第一个格点中心 Y
            "grid_w": 90,   # 格子横向间距
            "grid_h": 88,   # 格子纵向间距
            "threshold": 0.2 # 匹配阈值 (0.0-1.0)
        }
        # 3. 初始化模型
        self.classifier = None
        if model_path:
            model_abs = model_path
            self.classifier = PieceClassifier(model_abs)
            if os.path.exists(model_abs):
                try:
                    self.classifier = PieceClassifier(model_abs)
                    print(f"成功加载 CNN 模型: {model_abs}")
                except Exception as e:
                    print(f"错误: 加载模型时发生异常: {e}。将降级使用模板匹配。")
                    self.classifier = None
            else:
                print(f"警告: 模型文件不存在: {model_abs}，将降级使用模板匹配。")
        else:
            print("警告: 未配置模型路径，使用模板匹配。")

        
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """预加载 14 个棋子模板"""
        if not os.path.exists(self.template_dir):
            print(f"错误: 找不到模板目录 {self.template_dir}")
            return

        for file in os.listdir(self.template_dir):
            if file.endswith(".png"):
                name = file.split(".")[0] # 提取拼音文件名如 r_che
                path = os.path.join(self.template_dir, file)
                # 使用彩色模式读取，因为棋子颜色是重要特征
                img = cv2.imread(path)
                if img is not None:
                    self.templates[name] = img
        
        print(f"成功加载 {len(self.templates)} 个棋子模板。")

    def scan_board(self, img_path):
        """
        扫描全盘 90 个格点，返回 10x9 的二维列表
        """
        screen_img = cv2.imread(img_path)
        # if screen_img is not None:
        #     # 创建一个窗口并显示
        #     cv2.imshow('Chess Board Debug', screen_img)
            
        #     # 必须等待键盘输入，否则窗口会闪退
        #     print("按任意键关闭窗口...")
        #     cv2.waitKey(0) 
        #     cv2.destroyAllWindows()
        # else:
        #     print("图像读取失败，请检查路径。")
        # 初始化 10行 9列 的空棋盘 
        board = [["" for _ in range(9)] for _ in range(10)]
        
        for row in range(10):
            for col in range(9):
                # 1. 计算当前网格交叉点的像素中心坐标
                cx = self.grid_params["start_x"] + col * self.grid_params["grid_w"]
                cy = self.grid_params["start_y"] + row * self.grid_params["grid_h"]
                print(f"扫描格点 ({row}, {col}) 对应像素坐标 ({cx}, {cy})")
                # 2. 定义格点周围的采样区域 (ROI)
                # 采样半径设为格子大小的一半左右，确保覆盖完整棋子
                size = 45 
                x1, y1 = int(cx - size), int(cy - size)
                x2, y2 = int(cx + size), int(cy + size)
                
                # 防止坐标越界
                h, w = screen_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                roi = screen_img[y1:y2, x1:x2]
                # 弹出窗口显示当前格子的内容
                # cv2.imshow("Current Grid ROI", roi)
                # key = cv2.waitKey(0) & 0xFF 

                h, w = roi.shape[:2]
                print(f"ROI 尺寸: 宽 {w} 高 {h}")
                # 如果按下 Esc 键 (ASCII 27) 或 'q' 键，关闭窗口
                # if key == 27 or key == ord('q'):
                #     cv2.destroyAllWindows()
                
                # if roi.size == 0:
                #     continue

                # 3. 对该区域进行识别
                if self.classifier:
                    piece_name, confidence = self.classifier.predict(roi)
                    print(f"使用 CNN 识别结果: {piece_name} (置信度: {confidence:.2f})")
                    if confidence >= self.grid_params["threshold"]:
                        board[row][col] = piece_name
                    else:
                        board[row][col] = "None"
                else:
                    # 降级使用模板匹配
                    best_match = "None"
                    max_val = 0
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    for name, temp in self.templates.items():
                        temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                        
                        # 确保尺寸一致
                        if temp_gray.shape != roi_gray.shape:
                            temp_gray = cv2.resize(temp_gray, (roi_gray.shape[1], roi_gray.shape[0]))
                        res = cv2.matchTemplate(roi_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
                        _, val, _, _ = cv2.minMaxLoc(res)
                        
                        if val > max_val:
                            max_val = val
                            best_match = name
                    print(f"模板匹配得分最高: {max_val:.3f}, 模板: {best_match}")
                    if max_val >= self.grid_params["threshold"]:
                        board[row][col] = best_match
                    else:
                        board[row][col] = "None"                
        return board
    def scan_board_optimized(self, img_path):
        screen_img = cv2.imread(img_path)
        # 初始化 10行 9列 的空棋盘 
        board = [["" for _ in range(9)] for _ in range(10)]
        conf_matrix = [[0.0 for _ in range(9)] for _ in range(10)]
        
        for row in range(10):
            for col in range(9):
                cx = self.grid_params["start_x"] + col * self.grid_params["grid_w"]
                cy = self.grid_params["start_y"] + row * self.grid_params["grid_h"]
                
                # 1. 获取局部 ROI
                size = 45
                roi = screen_img[int(cy-size):int(cy+size), int(cx-size):int(cx+size)]
                if roi.size == 0: continue

                # 2. 判断颜色倾向 (基于 HSV 空间更准)
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 定义红色的两个区间（红色在 HSV 中跨越 0 度）
                mask_r1 = cv2.inRange(hsv_roi, np.array([0, 70, 50]), np.array([10, 255, 255]))
                mask_r2 = cv2.inRange(hsv_roi, np.array([170, 70, 50]), np.array([180, 255, 255]))
                mask_red = cv2.add(mask_r1, mask_r2)
                
                # 定义黑色的区间（主要看亮度 V 和 饱和度 S）
                mask_black = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 50]))

                # 计算像素点数量
                red_count = cv2.countNonZero(mask_red)
                print(f"格点 ({row}, {col}) 红色像素点数量: {red_count}")
                black_count = cv2.countNonZero(mask_black)
                print(f"格点 ({row}, {col}) 黑色像素点数量: {black_count}")

                # 3. 根据颜色选择匹配范围
                target_templates = []
                if red_count > 1000: # 这里的 50 是阈值，表示红色像素点足够多
                    target_templates = [k for k in self.templates.keys() if k.startswith('r_')]
                elif black_count > 500:
                    print(f"格点 ({row}, {col}) 识别为黑色棋子")
                    target_templates = [k for k in self.templates.keys() if k.startswith('b_')]
                    print(f"可选模板: {target_templates}")
                else:
                    board[row][col] = "None"
                    continue

                # 4. 灰度匹配
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                best_match = "None"
                max_val = 0
                for name in target_templates:
                    temp = self.templates[name]
                    temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                    
                    # 确保尺寸一致
                    if temp_gray.shape != roi_gray.shape:
                        temp_gray = cv2.resize(temp_gray, (roi_gray.shape[1], roi_gray.shape[0]))
                    res = cv2.matchTemplate(roi_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
                    _, val, _, _ = cv2.minMaxLoc(res)
                    
                    if val > max_val:
                        max_val = val
                        best_match = name
                    print(f"匹配得分: {val:.3f},匹配模板: {name}")
                if max_val >= self.grid_params["threshold"]:
                    board[row][col] = best_match
                    conf_matrix[row][col] = max_val
                else:
                    board[row][col] = "None"
                    conf_matrix[row][col] = max_val
                print(f"格点 ({row}, {col}) 最终识别结果: {board[row][col]} (得分: {max_val:.3f})")
        # 调用生成大图
        # self.generate_debug_image(img_path, board, conf_matrix)    
        return board

    def print_board(self, board):
        """格式化打印棋盘矩阵，方便调试"""
        print("\n--- 当前识别棋盘状态 ---")
        for row in board:
            # 将 None 简化为 '.' 增加可读性
            row_display = [p if p != "None" else " . " for p in row]
            print(" ".join(f"{p:8}" for p in row_display))
        print("-----------------------\n")

    def generate_debug_image(self, img_path, board_matrix, conf_matrix):
    
        """
        img_path: 原始截图路径
        board_matrix: scan_board 得到的 10x9 棋子名称矩阵
        conf_matrix: 扫描过程中记录的 10x9 置信度矩阵
        """
        # 1. 读取原图并创建副本
        debug_img = cv2.imread(img_path)
        if debug_img is None:
            print("无法读取图片进行绘图")
            return

        # 设置字体和颜色
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for row in range(10):
            for col in range(9):
                # 计算格点中心坐标
                cx = self.grid_params["start_x"] + col * self.grid_params["grid_w"]
                cy = self.grid_params["start_y"] + row * self.grid_params["grid_h"]
                
                piece_name = board_matrix[row][col]
                conf_val = conf_matrix[row][col]
                
                # 2. 绘制格点中心参考圆（红色代表识别为空，绿色代表识别到棋子）
                color = (0, 255, 0) if piece_name != "None" else (0, 0, 255)
                cv2.circle(debug_img, (int(cx), int(cy)), 5, color, -1)
                
                # 3. 绘制识别区域的外框 (ROI 范围)
                size = 45
                cv2.rectangle(debug_img, (int(cx-size), int(cy-size)), 
                            (int(cx+size), int(cy+size)), (255, 255, 0), 1)

                # 4. 标注识别信息
                if piece_name != "None":
                    # 显示棋子拼音前缀 (如 r_che -> che)
                    short_name = piece_name.split('_')[-1]
                    text = f"{short_name}:{conf_val:.2f}"
                    cv2.putText(debug_img, text, (int(cx-40), int(cy-50)), 
                                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # 如果是空位且置信度过高（误报），用黄色显示最高得分
                    if conf_val > 0.1:
                        cv2.putText(debug_img, f"{conf_val:.2f}", (int(cx-20), int(cy+55)), 
                                    font, 0.3, (0, 255, 255), 1)

        # 5. 保存并展示大图
        output_path = "assets/debug_full_board.png"
        cv2.imwrite(output_path, debug_img)
        print(f"对比大图已生成至: {output_path}")
        
        # 如果在桌面环境下，可以直接弹窗查看
        cv2.imshow("Full Board Debug", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def matrix_to_fen(self, board, side_to_move='w'):
        """
        将 10x9 矩阵转换为 FEN 字符串
        side_to_move: 'w' 代表红方先手, 'b' 代表黑方先手
        """
        # 建立映射表
        mapping = {
            "r_che": "R", "r_ma": "N", "r_xiang": "B", "r_shi": "A", "r_shuai": "K", "r_pao": "C", "r_bing": "P",
            "b_che": "r", "b_ma": "n", "b_xiang": "b", "b_shi": "a", "b_jiang": "k", "b_pao": "c", "b_zu": "p",
            "None": None, "": None  # 处理空位
        }
        
        fen_rows = []
        for row in board:
            empty_count = 0
            row_str = ""
            for cell in row:
                char = mapping.get(cell)
                if char:
                    # 如果之前有连续空位，先填入数字
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += char
                else:
                    empty_count += 1
            
            # 处理行末的空位
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)
        
        # 用 '/' 连接每一行，最后加上轮到谁走、回合数等信息（简化版）
        fen_base = "/".join(fen_rows)
        return f"{fen_base} {side_to_move} - - 0 1"
class PieceClassifier:
    def __init__(self, model_path, device=None):
        # 1. 自动选择设备
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 2. 模型文件存在性检查
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"棋子模型文件未找到: {model_path}。请确保文件存在，或在 PieceDetector 中传入 model_path=None 以禁用 CNN。")
        
        # 3. 加载模型存档
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # 3. 初始化模型结构 (需与训练时保持完全一致)
        # 如果是棋子模型使用 MobileNetV3
        self.model = models.mobilenet_v3_small()
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(self.class_names))
        
        # 4. 注入权重并设为评估模式
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 5. 定义预处理转换 (必须与训练阶段完全相同)
        self.transform = transforms.Compose([
            transforms.Resize((90, 90)), # 对应你训练时的 IMG_SIZE
            transforms.ToTensor(),
        ])

    def predict(self, roi_bgr):
        """
        输入: OpenCV 读取的 BGR 图片 (ROI)
        返回: 类别名称和置信度
        """
        # OpenCV BGR 转 PIL RGB
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # 预处理并增加 Batch 维度
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # 使用 Softmax 转换为概率分布
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probabilities, 0)
            
            return self.class_names[pred_idx.item()], conf.item()