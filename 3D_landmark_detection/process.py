import os
import sys
import shutil
import time
import subprocess
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# =========================================================================
#                          配置路径
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 关键文件路径
INPUT_CT_PATH = r"D:\study\HRNet-Facial-Landmark-Detection\data\ct/images/NIFTI/H014.nii.gz" # 请修改为您的输入路径
PREPROCESSED_TEMP_PATH = os.path.join(OUTPUT_DIR, "preprocessed_temp.nii.gz")
WEB_TARGET_FILENAME = "privacydata.nii.gz"
WEB_DATA_PATH = os.path.join(PROJECT_ROOT, WEB_TARGET_FILENAME) 
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "frontal_view.png")
CAPTURE_HTML_PATH = os.path.join(PROJECT_ROOT, "capture.html")

# 引入您的模块
sys.path.append(PROJECT_ROOT)
try:
    # 假设您的 preprocess 逻辑在这里
    from preprocess import facial_ct_preprocessing_pipeline
    from integrated_pipeline import run_pipeline 
    pass 
except ImportError:
    pass

# =========================================================================
#                          流程控制
# =========================================================================

def start_server():
    """在后台启动 SimpleHTTP Server"""
    print(f"   -> 启动后台 HTTP 服务器 (Port 8000)...")
    # 使用 Popen 不阻塞主线程
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc

def capture_screenshot_with_selenium():
    """使用 Selenium 智能截图"""
    
    # 1. 启动 HTTP Server
    server_proc = start_server()
    time.sleep(2) # 稍微等待服务器就绪

    driver = None
    try:
        # 2. 配置 Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless") # 无头模式
        chrome_options.add_argument("--window-size=1200,1200") # 必须设置大窗口
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument("--hide-scrollbars")
        # 解决某些环境下 WebGL 不工作的问题
        chrome_options.add_argument("--use-gl=swiftshader") 
        chrome_options.add_argument("--enable-webgl")
        
        print("   -> 启动 Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # 3. 访问页面
        target_url = "http://localhost:8000/capture.html"
        print(f"   -> 访问: {target_url}")
        driver.get(target_url)

        # 4. === 智能等待 (Smart Wait) ===
        # 等待 JS 里的 window.renderDone 变为 true
        print("   -> 等待 3D 渲染完成...")
        max_wait = 30 # 最多等 30 秒
        start_time = time.time()
        is_ready = False
        
        while time.time() - start_time < max_wait:
            # 执行 JS 获取状态
            is_ready = driver.execute_script("return window.renderDone;")
            if is_ready:
                break
            time.sleep(0.5)
            
        if not is_ready:
            print("❌ 错误: 渲染等待超时，截图可能为空。")

        # 5. 截图
        canvas = driver.find_element(By.ID, "gl")
        success = canvas.screenshot(SCREENSHOT_PATH)
        
        if success:
            print(f"✅ 截图已保存: {SCREENSHOT_PATH}")
            # 验证文件大小
            if os.path.exists(SCREENSHOT_PATH) and os.path.getsize(SCREENSHOT_PATH) > 1000:
                print(f"   (文件大小正常: {os.path.getsize(SCREENSHOT_PATH)} bytes)")
            else:
                print("⚠️ 警告: 截图文件过小，可能为空白。")
        else:
            print("❌ 截图操作返回失败。")

    except Exception as e:
        print(f"❌ Selenium 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        if driver:
            driver.quit()
        if server_proc:
            server_proc.kill()
            print("   -> HTTP Server 已关闭")

def main():
    print("="*60)
    print("步骤 1: 准备数据")
    # -----------------------------------------------------
    # 步骤 1: 预处理 (用于显示)
    # -----------------------------------------------------
    print("\n[1/3] 预处理 CT 数据 (用于生成3D视图)...")
    try:
        # 生成 0-255 的可视化文件
        facial_ct_preprocessing_pipeline(INPUT_CT_PATH, PREPROCESSED_TEMP_PATH)
        
        # 【关键动作】将文件复制到项目根目录，并重命名为 privacydata.nii.gz
        # 这完全模拟了之前的成功操作，确保网页能读到文件。
        if os.path.exists(WEB_DATA_PATH):
            os.remove(WEB_DATA_PATH) # 先清理旧文件
        shutil.copy(PREPROCESSED_TEMP_PATH, WEB_DATA_PATH)
        
        print(f"   -> 可视化数据已部署至: {WEB_DATA_PATH}")
        
    except Exception as e:
        print(f"❌ 步骤 1 失败: {e}")
        sys.exit(1)

    print("\n步骤 2: 截取正面视图")
    capture_screenshot_with_selenium()
    
    # 步骤 3: 后续处理 (Landmark Detection)
    if os.path.exists(SCREENSHOT_PATH) and os.path.getsize(SCREENSHOT_PATH) > 0:
         print("\n步骤 3: 运行关键点检测...")
         # 这里调用您的 integrated_pipeline
         try:
             from integrated_pipeline import run_pipeline
             # 假设 run_pipeline 接收: image_path, nii_path, output_dir
             run_pipeline(SCREENSHOT_PATH, WEB_DATA_PATH, OUTPUT_DIR)
         except Exception as e:
             print(f"无法运行后续管线: {e}")
    else:
        print("❌ 跳过步骤 3，因为截图失败。")

if __name__ == "__main__":
    main()

