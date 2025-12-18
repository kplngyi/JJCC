from jj_chess_bot.vision.piece_detector import PieceDetector
detector = PieceDetector()
# 读取你之前的截图进行测试
test_img_path = "/Users/hpyi/Hobby/JJCC/assets/screenshots/WeChat_JJ象棋_6610.png"
result_matrix = detector.scan_board(test_img_path)
detector.print_board(result_matrix)
# 2. 转换为 FEN
fen_str = detector.matrix_to_fen(result_matrix)
print(f"生成的 FEN: {fen_str}")