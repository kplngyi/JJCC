import os
from Quartz import (
    CGWindowListCopyWindowInfo, 
    kCGWindowListOptionAll, 
    kCGNullWindowID, 
    CGWindowListCreateImage, 
    kCGWindowImageDefault, 
    CGRectNull,
    kCGWindowListOptionIncludingWindow
)
from AppKit import NSBitmapImageRep, NSPNGFileType

def capture_process_screenshot(target_pid, output_dir="screenshots"):
    """
    捕获指定 PID 进程的所有窗口并保存为 PNG
    """
    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 获取系统中所有窗口的信息列表
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    print(f"系统中共有 {len(window_list)} 个窗口。")
    ## 将JJ留下
    captured_count = 0

    for window in window_list:
        # 检查窗口所属的进程 ID (PID)
        if window.get('kCGWindowOwnerPID') == target_pid:
            window_id = window.get('kCGWindowNumber')
            # 获取窗口名称，如果没有名称则显示为 Unknown
            window_name = window.get('kCGWindowName', 'Unknown')
            if window_name != "JJ象棋":
                continue
            print(f"正在捕获窗口: {window_name} (ID: {window_id})")
            # 获取窗口所属的应用名称
            owner_name = window.get('kCGWindowOwnerName', 'Process')

            # 2. 针对特定窗口 ID 创建图像快照
            # CGRectNull 表示自动根据窗口边界进行裁剪
            image = CGWindowListCreateImage(
                CGRectNull, 
                kCGWindowListOptionIncludingWindow, 
                window_id, 
                kCGWindowImageDefault
            )
            
            if image:
                # 3. 将 CGImage 转换为 PNG 并保存
                file_name = f"{owner_name}_{window_name}_{window_id}.png".replace("/", "-")
                file_path = os.path.join(output_dir, file_name)
                
                # 转换逻辑：CGImage -> NSBitmapImageRep -> PNG Data -> File
                bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(image)
                png_data = bitmap_rep.representationUsingType_properties_(NSPNGFileType, None)
                
                if png_data:
                    png_data.writeToFile_atomically_(file_path, True)
                    print(f"成功保存窗口: {window_name} (ID: {window_id}) -> {file_path}")
                    captured_count += 1

    return captured_count

if __name__ == "__main__":
    # 替换为你想要截取的进程 PID
    # 例如你图片中的微信小程序 PID: 31499
    TARGET_PID = 4076
    
    print(f"正在尝试截取 PID 为 {TARGET_PID} 的窗口...")
    count = capture_process_screenshot(TARGET_PID)
    
    if count == 0:
        print("未找到该进程的有效窗口。请检查 PID 是否正确，或程序是否拥有'屏幕录制'权限。")
    else:
        print(f"任务完成，共捕获 {count} 张图片。")