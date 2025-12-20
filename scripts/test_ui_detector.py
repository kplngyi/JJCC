from jj_chess_bot.capture.screenshot import ScreenCapture
from jj_chess_bot.vision.ui_detector import UIDetector
from jj_chess_bot.config.config_loader import config
from jj_chess_bot.vision.piece_detector import PieceDetector
from jj_chess_bot.engine.engine_wrapper import ChessEngine
from jj_chess_bot.control.operator import ChessOperator
import cv2
TARGET_PID = 4076  # 实际的 JJ象棋 进程 PID
# 调试代码逻辑
cap = ScreenCapture(TARGET_PID)
# 在进入循环前，判断一次是红还是黑
print("正在检测对局角色...")
img = cap.grab_by_name()
pos_windows = cap.get_window_bounds("JJ象棋")
operator = ChessOperator("Mini Program")
detector = UIDetector()
pos, conf = detector.find_button(img)
print(f"位置: {pos}, 置信度: {conf}")
operator.click_piece(pos_windows, pos[0], pos[1])
