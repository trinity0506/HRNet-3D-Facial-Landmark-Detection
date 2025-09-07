import asyncio
import pyppeteer
import os

async def main():
    # 启动 Chromium 浏览器
    browser = await pyppeteer.launch(headless=True)
    page = await browser.newPage()

    # 设置 HTML 文件的路径
    html_file_path = os.path.abspath('niivue.html')

    # 加载本地 HTML 文件
    await page.goto(f'file://{html_file_path}')

    # 等待页面渲染完成
    await page.waitForSelector('#gl')

    # 等待场景保存完成
    await page.waitForFunction(
        "() => { return document.getElementById('gl').getAttribute('data-saved') === 'true'; }",
        timeout=120000  # 增加到 120 秒
    )

    # 截取页面的指定区域并保存为图像
    await page.screenshot({'path': 'face_frontal_view.png', 'clip': {'x': 0, 'y': 0, 'width': 640, 'height': 640}})

    # 关闭浏览器
    await browser.close()

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
