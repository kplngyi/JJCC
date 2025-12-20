import pyautogui
import time
import random
from AppKit import NSWorkspace

class ChessOperator:
    """
    负责将逻辑走法转化为macOS上的物理鼠标操作
    """
    def __init__(self, target_name, scale = 1.0):
        self.target_name = target_name
        self.scale = scale
        # 初始安全设置
        pyautogui.PAUSE = 0.1 # 默认停顿
        pyautogui.FAILSAFE = True # 
    def focus_window(self):
        """置顶窗口，确保点击不落空"""
        apps = NSWorkspace.sharedWorkspace().runningApplications()
        for app in apps:
            if self.target_name in (app.localizedName() or ""):
                app.activateWithOptions_(1)
                time.sleep(0.3)
                return True
        return False
    def _convert_coords(self, window_pos, rel_x, rel_y):
        """内部方法：截图像素坐标 -> 屏幕逻辑坐标"""
        print(rel_x,rel_y)
        abs_x = window_pos['x'] + rel_x
        abs_y = window_pos['y'] + rel_y
        print(abs_x,abs_y)
        return abs_x, abs_y

    def click_piece(self, window_pos, rel_x, rel_y):
        """点击棋盘上的特定像素点"""
        if not self.focus_window():
            print(self.target_name + "未找到")
            pyautogui.moveTo(window_pos['x'], window_pos['y'], duration=random.uniform(0.2, 0.4))

        abs_x, abs_y = self._convert_coords(window_pos, rel_x, rel_y)
        
        # 拟人性：加入微小随机偏移
        target_x = abs_x + random.uniform(-1, 1)
        target_y = abs_y + random.uniform(-1, 1)

        pyautogui.moveTo(target_x, target_y, duration=random.uniform(0.2, 0.4))
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.12))
        pyautogui.mouseUp()
    def get_move_pixels(self, cfg, move_str):
        """
        将 FEN 走法（如 'h2e2'）解析为起始和终止的像素坐标
        :param move_str: 引擎返回的 4 位走法字符串
        :return: ((start_x, start_y), (end_x, end_y))
        """
        # 内部辅助函数处理单个坐标点（如 'h2'）
        if move_str == 'none': # 特殊处理：认输
            print("认输")
            return None, None

        def parse_pos(pos):
            col_char = pos[0] # 'h'
            row_char = pos[1] # '2'
            
            col_idx = ord(col_char) - ord('a') # 'a'->0, 'h'->7
            row_idx = int(row_char)            # '2'->2

            
            # 计算像素位置
            # X = 起点 + 列索引 * 格子宽度
            # Y = 起点 + (9 - 行索引) * 格子高度 (因为 y 轴向下，a9 在最上方)
            px = cfg['start_x'] + col_idx * cfg['cell_w']
            py = cfg['start_y'] + (9 - row_idx) * cfg['cell_h']
            
            return int(px), int(py)

        start_px = parse_pos(move_str[:2])
        end_px = parse_pos(move_str[2:])
        return start_px, end_px 

    def execute_move(self, window_pos, start_px, end_px):
        """执行完整的棋局走步"""
        self.click_piece(window_pos, start_px[0], start_px[1])
        time.sleep(random.uniform(0.4, 0.7))
        self.click_piece(window_pos, end_px[0], end_px[1])


