import os
import yaml

class Config:
    def __init__(self):
        # 1. 定位项目根目录 (Project Root)
        # 此脚本在 src/jj_chess_bot/config/ 下，向上三级即为根目录
        current_file = os.path.abspath(__file__)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        print(f"项目根目录: {self.root_dir}")
        
        # 2. 加载 YAML 文件
        config_path = os.path.join(self.root_dir, "src/jj_chess_bot/config/config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get_abs_path(self, key_path):
        """
        根据 YAML 中的相对路径获取绝对路径
        key_path 示例: ['engine', 'path']
        """
        # 逐级获取配置，例如 self.data['engine']['path']
        val = self.data
        for key in key_path:
            val = val[key]
        
        # 拼接并返回绝对路径
        return os.path.join(self.root_dir, val)

# 实例化单例供全项目使用
config = Config()