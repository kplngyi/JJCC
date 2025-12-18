from jj_chess_bot.vision.piece_detector import PieceDetector
detector = PieceDetector()
results = detector.find_pieces("test_screen.png", threshold=0.8)
for res in results:
    print(f"Found {res['name']} at {res['pos']} with confidence {res['confidence']:.2f}")