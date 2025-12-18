from jj_chess_bot.capture.screenshot import ScreenCapture
TARGET_PID = 4076  # 实际的 JJ象棋 进程 PID
cap = ScreenCapture(TARGET_PID)
img = cap.grab()
img.save("test_screen.png")
print("✅ Screenshot saved as test_screen.png")