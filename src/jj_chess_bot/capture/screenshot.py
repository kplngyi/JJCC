from mss import mss
from PIL import Image
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGNullWindowID
from pathlib import Path
import os
import io
from Quartz import (
    CGWindowListCopyWindowInfo, 
    kCGWindowListOptionAll, 
    kCGNullWindowID, 
    CGWindowListCreateImage, 
    kCGWindowImageDefault, 
    CGRectNull,
    kCGWindowListOptionIncludingWindow,
    kCGWindowListOptionOnScreenOnly
)
from AppKit import NSBitmapImageRep, NSPNGFileType

class ScreenCapture:
    def __init__(self,target_pid,output_dir="assets/screenshots"):

        self.target_pid = target_pid

        # 当前文件路径
        current_file = Path(__file__).resolve()

        # 项目根目录（src/ 在项目根目录下）
        project_root = current_file.parent.parent.parent.parent  # screenshot.py -> capture -> jj_chess_bot -> src -> 项目根

        # 将相对路径与项目根目录拼接，确保路径永远一致
        self.output_dir = os.path.join(project_root, output_dir)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建目录: {self.output_dir}")
        print(f"图片将保存至实际路径: {os.path.abspath(self.output_dir)}")
    def grab(self, target_name="JJ象棋"):
        """
         grab the screenshot which contains the target_name window
        """
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
        # print(f"系统中共有 {len(window_list)} 个窗口。")
        for window in window_list:
            # print(window)
            if window.get('kCGWindowOwnerPID') == self.target_pid:
                window_id = window.get('kCGWindowNumber')
                window_name = window.get('kCGWindowName', 'Unknown')
                if window_name != target_name:
                    continue
                owner_name = window.get('kCGWindowOwnerName', 'Process')

                # make screenshot
                cg_image = CGWindowListCreateImage(
                    CGRectNull, 
                    kCGWindowListOptionIncludingWindow, 
                    window_id, 
                    kCGWindowImageDefault
                )
                if cg_image:
                    # CGImage to PIL.Image
                    # print(f"Capturing window: {window_name} (ID: {window_id})")
                    bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
                    png_data = bitmap_rep.representationUsingType_properties_(NSPNGFileType, None)
                    if png_data:
                        file_name = f"{owner_name}_{window_name}_{window_id}.png".replace("/", "-")
                        file_path = os.path.join(self.output_dir, file_name)
                        png_data.writeToFile_atomically_(file_path, True)
                        # print(f"Saved window: {window_name} (ID: {window_id}) -> {file_path}")
                        img = Image.open(file_path)
                        return img
        return None
    
    def grab_by_name(self, target_window_name="JJ象棋"):
        """
        通过窗口标题名称截取窗口内容，不依赖 PID，不经过硬盘
        """
        # 1. 获取所有在屏幕上的窗口信息
        # kCGWindowListOptionOnScreenOnly 过滤掉后台隐藏窗口，提升搜索速度
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

        for window in window_list:
            # 获取窗口标题和所属进程名
            w_name = window.get('kCGWindowName', '')
            owner_name = window.get('kCGWindowOwnerName', '')

            # 匹配逻辑：窗口标题包含目标名称，或者进程名包含目标名称
            if target_window_name in w_name or target_window_name in owner_name:
                window_id = window.get('kCGWindowNumber')
                
                # 2. 创建窗口图像
                cg_image = CGWindowListCreateImage(
                    CGRectNull, 
                    kCGWindowListOptionIncludingWindow, 
                    window_id, 
                    kCGWindowImageDefault
                )
                
                if cg_image:
                    # 3. 内存转换：CGImage -> NSBitmap -> PIL Image
                    # 避开了文件读写，极大提升识别频率
                    bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
                    png_data = bitmap_rep.representationUsingType_properties_(NSPNGFileType, None)
                    
                    if png_data:
                        # 使用 io.BytesIO 在内存中打开图片
                        return Image.open(io.BytesIO(png_data.bytes()))
        
        print(f"⚠️ 未找到名称包含 '{target_window_name}' 的窗口")
        return None