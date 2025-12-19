import time
from jj_chess_bot.config.config_loader import config
from jj_chess_bot.capture.screenshot import ScreenCapture
from jj_chess_bot.vision.piece_detector import PieceDetector
from jj_chess_bot.engine.engine_wrapper import ChessEngine
from jj_chess_bot.control.operator import ChessOperator

def main():
    # 1. 初始化所有组件
    detector = PieceDetector()
    engine = ChessEngine(config.get_abs_path(['engine', 'path']))
    operator = ChessOperator(config.get_target_name())
    # a9
    cfg = config.get_board_config()
    last_move_fen = ""
    # return 
    print("🚀 Bot 已启动，进入全自动对弈模式...")

    while True:
        # 1. 截取屏幕并识别棋盘
        # 建议使用内存截图而不是保存文件，速度更快
        # screenshot_path = "assets/screenshots/current_screen.png" 
        # 这里你可以调用你的截屏工具函数
        TARGET_PID = 4076  # 实际的 JJ象棋 进程 PID
        cap = ScreenCapture(TARGET_PID)
        img = cap.grab_by_name()
        board_matrix = detector.scan_board(img)
        detector.print_board(board_matrix)
        # 2. 转换为 FEN (假设你轮到红方 w)
        current_fen = detector.matrix_to_fen(board_matrix, side_to_move='w')
        print(f"当前 FEN: {current_fen}")
        # time.sleep(22)
        # # 3. 判断是否轮到我走
        # # 简单的逻辑：如果棋盘状态没变，说明对手还没走完
        # if current_fen == last_move_fen:
        #     print("⏳ 等待对手走棋...", end="\r")
        #     time.sleep(2)
        #     continue
            
        # print(f"\n检测到新局面，开始计算...")
        
        # 4. 获取最佳走法
        best_move = engine.get_best_move(current_fen)
        time.sleep(30)
        
        if best_move:
            # 5. 执行物理点击
            # operator.execute_move(best_move)
            
            # 更新状态，防止重复走同一步
            # 注意：实际操作后棋盘会变，这里记录一下
            last_move_fen = current_fen 
            
        time.sleep(1) # 给动画一点缓冲时间

if __name__ == "__main__":
    main()