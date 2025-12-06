import sys
import time
import base64
# Try importing PyQt5, handle installation errors if necessary
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtCore import QUrl
except ImportError:
    print("Error: PyQt5 or QtWebEngineWidgets not found. Please install via 'pip install PyQt5 PyQtWebEngine'")
    sys.exit(1)

def save_canvas_from_url(url, out_path):
    app = QApplication(sys.argv)
    view = QWebEngineView()
    
    # Setup callback or simple delay mechanism? 
    # Since this is a CLI tool, we will load, wait, execute JS, save, quit.
    
    view.load(QUrl(url))
    view.show()

    # Helper to run sync-ish JS (with event loop processing)
    def js_eval(page, script, timeout=5.0):
        result = [None]
        finished = [False]
        def cb(r):
            result[0] = r
            finished[0] = True
        
        page.runJavaScript(script, cb)
        
        start = time.time()
        while not finished[0]:
            app.processEvents()
            if time.time() - start > timeout:
                break
        return result[0]

    # Wait for page load
    # Simple busy wait checking for ready
    print("Loading page:", url)
    
    # Initial wait for network/DOM
    # More robust: check for a variable on window
    ready = False
    for i in range(20): # wait up to 10s
        app.processEvents()
        time.sleep(0.5)
        # Check if Niivue is attached
        is_loaded = js_eval(view.page(), "!!window.nv && !!window.nv.volumes && window.nv.volumes.length > 0")
        if is_loaded:
            ready = True
            break
    
    if not ready:
        print("警告：等待 Niivue 就绪超时，仍尝试截图。")

    # -----------------------
    # 核心修改部分开始
    # -----------------------
    js_force_3d = r"""
    (function(){
    try {
        // 1. 强制固定视角 (原逻辑)
        var az = 180, el = 0;
        if (window.prepareFrontView && typeof window.prepareFrontView === 'function') {
            try { window.prepareFrontView(az, el); } catch(e) {}
        } else if (window.nv) {
            if (typeof window.nv.setRenderAzimuthElevation === 'function') {
                window.nv.setRenderAzimuthElevation(az, el);
            } else if (typeof window.nv.setRotationDegrees === 'function') {
                window.nv.setRotationDegrees(0, az, 0);
            }
        }

        // 2. [新增] 标准化缩放逻辑：抵消 Auto-fit 带来的比例差异
        // 目标：让 1mm 在屏幕上总是对应固定的像素数
        if (window.nv && window.nv.volumes && window.nv.volumes.length > 0) {
            var vol = window.nv.volumes[0];
            // 获取头文件信息
            var dims = vol.hdr.dims;    // [dim_idx, x, y, z, t...]
            var pix = vol.hdr.pixDims;  // [unused, x, y, z...]
            
            // 计算xyz三个方向的物理全长 (voxels * mm/voxel)
            // 注意：NIfTI dims 通常从索引 1 开始对应 x,y,z
            var dimX = dims[1] * pix[1];
            var dimY = dims[2] * pix[2];
            var dimZ = dims[3] * pix[3];
            
            // 找出最大物理边长（Niivue Auto-fit 通常基于最大边长归一化）
            var maxPhysDim = Math.max(dimX, dimY, dimZ);
            
            // 设定缩放公式：
            // Niivue 的默认行为近似于：Scale_Applied = 1.0 (Fit) / maxPhysDim
            // 我们希望 Scale_Final = Const
            // 所以我们需要设置 nv.setScale( coeff * maxPhysDim )
            // 200.0 是一个参考基准常数（假设标准头长200mm），你可以调整这个除数来改变全局缩放大小
            
            if (typeof window.nv.setScale === 'function') {
                // 这里的逻辑是：如果物体越大(maxPhysDim大)，Niivue默认会自动缩得越小。
                // 为了保持比例一致，我们需要把Scale设得和物体尺寸成正比。
                // 具体数值可能需要根据 Niivue 版本微调，但线性关系是关键。
                window.nv.setScale(maxPhysDim / 200.0);
            }
        }

        // 3. 强制重绘
        if (window.nv && typeof window.nv.drawScene === 'function') {
            for (var i = 0; i < 8; i++) {
                try { window.nv.drawScene(); } catch(e){ console.warn(e); }
            }
        }
        
        document.title = 'niivue-ready';
        return true;
    } catch (err) {
        console.warn('force3d error', err);
        return false;
    }
    })();
    """
    # -----------------------
    # 核心修改部分结束
    # -----------------------

    # Execute logic
    view.page().runJavaScript(js_force_3d)
    
    # Wait a bit for rendering to settle (GPU)
    for i in range(10): # 2 seconds allow for heavy rendering
        app.processEvents()
        time.sleep(0.2)
        
    # Secondary draw call safety
    try:
        view.page().runJavaScript("if(window.nv && typeof window.nv.drawScene === 'function'){ for(var i=0;i<6;i++){ window.nv.drawScene(); } }")
    except Exception:
        pass
    time.sleep(0.5)

    # Get canvas data
    data_url = js_eval(view.page(), """
    (function(){
      var c = document.getElementById('gl');
      if(!c) return null;
      try { return c.toDataURL('image/png'); } catch(e) { return null; }
    })();
    """, timeout=5.0)

    if not data_url:
        print("Error: Cannot get canvas data (WebGL context might not be ready or id='gl' missing).")
        view.close()
        app.quit()
        sys.exit(1)

    header, b64 = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64)
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    print("Saved image to:", out_path)

    view.close()
    app.quit()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python capture_front_image.py <url> <out.png>")
        sys.exit(1)
    url = sys.argv[1]
    out = sys.argv[2]
    save_canvas_from_url(url, out)

# windows : $p = Start-Process -FilePath python -ArgumentList '-m http.server 8000' -WorkingDirectory 'D:\study\HRNet-Facial-Landmark-Detection' -PassThru; python .\3D_landmark_detection\capture_front_image.py "http://localhost:8000/capture.html" out.png; $p.Kill()
#(cd "D:/study/HRNet-Facial-Landmark-Detection" && python3 -m http.server 8000 >/dev/null 2>&1 & pid=$!; python3 .\3D_landmark_detection\capture_front_image.py "http://localhost:8000/capture.html" out.png; kill $pid)