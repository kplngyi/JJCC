关于 Retina 屏幕的“陷阱”
在 Mac 上，你可能会发现一个奇怪的现象：
你的窗口尺寸报告是 419x788。
但如果你打开保存的 test_screen.png 查看属性，它的分辨率可能是 838x1576（正好是 2 倍）。
## 截图问题
这是一个非常典型的 macOS 窗口阴影与焦点处理机制 导致的问题。在 macOS 中，当一个窗口成为“当前活动窗口”（Active Window）时，系统会为其渲染一个更宽、更深、更亮的外发光阴影。许多截图库（如 pyautogui 或 mss 的某些配置）在截取窗口时，会把这些“阴影像素”也算进去。1. 为什么分辨率会改变？你的两组数据正好印证了这一点：$1062 \times 1800$ (活动状态)：包含了完整的高亮阴影。$974 \times 1712$ (非活动状态)：阴影变窄或消失，导致总像素减少。核心矛盾： 你的 grid_params（如 start_x: 123）是从图片的左上角（包含阴影的边缘）开始算的。由于阴影宽度变了，棋盘相对于图片左上角的“位移”也变了，导致坐标全部对不准。
## 实现整体逻辑
![alt text](image.png)
### 继续挑战
![alt text](image-1.png)

### 
--- 当前识别棋盘状态 ---
none     b_che    b_xiang  b_shi    b_jiang  b_shi    b_xiang  b_ma     b_che   
none     none     none     none     none     none     none     none     none    
none     b_pao    b_ma     none     none     none     none     none     none    
b_zu     none     b_zu     none     none     none     b_pao    r_ma     b_zu    
none     none     none     none     r_pao    none     none     none     none    
none     none     none     none     none     none     b_zu     none     none    
r_bing   none     r_bing   none     r_bing   none     none     none     r_bing  
none     none     none     none     none     none     none     none     r_ma    
none     none     none     none     none     none     none     none     none    
r_che    r_ma     r_xiang  r_shi    r_shuai  r_shi    r_xiang  none     r_che  
![alt text](image-2.png)