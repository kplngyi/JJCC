from jj_chess_bot.vision.piece_detector import PieceDetector
detector = PieceDetector()
# 读取你之前的截图进行测试
test_img_path = "/Users/hpyi/Hobby/JJCC/assets/screenshots/AIRWeChat_JJ象棋_2802.png"
result_matrix = detector.scan_board_optimized(test_img_path)
detector.print_board(result_matrix)
