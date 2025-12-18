import subprocess
import os
import signal
from jj_chess_bot.config.config_loader import config

class ChessEngine:
    def __init__(self, engine_path, depth=15, movetime=2000):
        """
        初始化引擎
        :param engine_path: 引擎二进制文件的绝对路径
        :param depth: 搜索深度
        :param movetime: 每次思考的时间限制 (毫秒)
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"找不到引擎文件: {engine_path}")
            
        self.engine_path = engine_path
        self.depth = depth
        self.movetime = movetime
        self.process = None
        self.start_engine()

    def start_engine(self):
        """启动引擎进程并进入 UCCI 模式"""
        # 确保文件有执行权限
        print(f"启动引擎: {self.engine_path}")
        os.chmod(self.engine_path, 0o755)
        
        self.process = subprocess.Popen(
            self.engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲
            # 在 Unix 系统下，这有助于防止僵尸进程
            preexec_fn=os.setsid if os.name != 'nt' else None 
        )
        
        # UCCI 协议初始化指令
        self._send("isready")
        
        # 等待引擎准备就绪
        while True:
            line = self.process.stdout.readline().strip()
            if line == "readyok":
                print("✅ 引擎 (UCCI) 已就绪")
                break

    def _send(self, command):
        print(f"➡️ 发送指令: {command}")
        """向引擎标准输入发送指令"""
        if self.process and self.process.stdin:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()

    def get_best_move(self, fen):
        print(f"♟️ 计算最佳走法 for FEN: {fen}")
        """
        输入 FEN 字符串，获取引擎计算的最优走法
        :param fen: 格式如 "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
        :return: 走法字符串，如 "h2e2"
        """
        # 1. 设置局面
        self._send(f"position fen {fen}")
        
        # 2. 开始搜索
        # 可以按深度搜: f"go depth {self.depth}"
        # 或按时间搜: f"go movetime {self.movetime}"
        self._send(f"go depth {self.depth}")
        
        # 3. 解析输出直到找到 bestmove
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                # line 示例: "bestmove h2e2 ponder h9g7"
                parts = line.split()
                if len(parts) >= 2:
                    move = parts[1]
                    print(f"🤖 引擎思考结果: {move}")
                    return move
            elif "error" in line.lower():
                print(f"❌ 引擎报错: {line}")
                return None

    def quit(self):
        print("👋 引擎已关闭")
        """安全关闭引擎"""
        if self.process:
            self._send("quit")
            self.process.terminate()
            print("👋 引擎已关闭")

    def __del__(self):
        self.quit()

# --- 简单测试逻辑 ---
if __name__ == "__main__":
    # 这里的路径仅供测试，实际应从 config.yaml 加载
    # 自动获取引擎的绝对路径
    ENGINE_PATH = config.get_abs_path(['engine', 'path'])
    # 初始开局 FEN
    START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    
    try:
        engine = ChessEngine(ENGINE_PATH)
        move = engine.get_best_move(START_FEN)
        print(f"测试走法结果: {move}")
    except Exception as e:
        print(f"发生错误: {e}")