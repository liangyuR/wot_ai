"""
打包成单个 exe - 最简单的方案
只需要一个 exe 文件，双击运行，配置和录制都在一个程序里
"""
import subprocess
import sys
import shutil
from pathlib import Path
import zipfile


def install_pyinstaller():
    """安装 PyInstaller"""
    print("=" * 80)
    print("📦 检查 PyInstaller...")
    print("=" * 80)
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
        return True
    except ImportError:
        print("⚠️  PyInstaller 未安装，正在安装...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyinstaller"],
                check=True
            )
            print("✓ PyInstaller 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ PyInstaller 安装失败: {e}")
            return False


def build_single_exe():
    """打包成单个 exe"""
    print("\n" + "=" * 80)
    print("📦 打包成单个 exe")
    print("=" * 80)
    print()
    print("这可能需要 3-5 分钟，请耐心等待...")
    print()
    
    # 获取绝对路径
    base_dir = Path(__file__).parent.absolute()
    config_file = base_dir / "wot_client" / "configs" / "client_config.yaml"
    data_collection_dir = base_dir / "wot_client" / "data_collection"
    utils_dir = base_dir / "wot_client" / "utils"
    gui_file = base_dir / "wot_client" / "config_gui.py"
    
    # 验证文件存在
    if not config_file.exists():
        print(f"✗ 配置文件不存在: {config_file}")
        return False
    if not data_collection_dir.exists():
        print(f"✗ data_collection 目录不存在: {data_collection_dir}")
        return False
    if not utils_dir.exists():
        print(f"✗ utils 目录不存在: {utils_dir}")
        return False
    if not gui_file.exists():
        print(f"✗ GUI 文件不存在: {gui_file}")
        return False
    
    print(f"✓ 找到配置文件: {config_file}")
    print(f"✓ 找到 data_collection: {data_collection_dir}")
    print(f"✓ 找到 utils: {utils_dir}")
    print()
    
    cmd = [
        "pyinstaller",
        "--name=坦克世界AI数据采集工具",
        "--onefile",  # 单个 exe 文件
        "--console",  # 显示控制台（方便看录制日志）
        "--icon=NONE",
        "--clean",
        "--distpath=dist",
        "--workpath=build/temp",
        "--specpath=build",
        # 添加数据文件（使用绝对路径）
        f"--add-data={config_file};configs",
        # 添加代码模块（使用绝对路径）
        f"--add-data={data_collection_dir};data_collection",
        f"--add-data={utils_dir};utils",
        # 隐藏导入 - Windows 相关
        "--hidden-import=win32timezone",
        "--hidden-import=win32api",
        "--hidden-import=win32con",
        "--hidden-import=pywintypes",
        "--hidden-import=win32gui",
        # 隐藏导入 - 核心依赖
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=mss",
        "--hidden-import=mss.windows",
        "--hidden-import=pynput",
        "--hidden-import=pynput.keyboard",
        "--hidden-import=pynput.mouse",
        "--hidden-import=loguru",
        "--hidden-import=yaml",
        "--hidden-import=tkinter",
        "--hidden-import=psutil",
        # OpenCV 子模块
        "--hidden-import=numpy.core._methods",
        "--hidden-import=numpy.lib.format",
        # 收集所有相关包
        "--collect-submodules=cv2",
        "--collect-binaries=cv2",
        "--collect-submodules=numpy",
        "--collect-submodules=mss",
        "--collect-submodules=pynput",
        "--collect-submodules=loguru",
        # 主程序（使用绝对路径）
        str(gui_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print()
        print("✓ 打包成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 打包失败")
        print(f"错误输出: {e.stderr}")
        return False


def create_release_package():
    """创建发布包"""
    print("\n" + "=" * 80)
    print("📦 创建发布包")
    print("=" * 80)
    
    # 创建发布目录
    release_dir = Path("dist/WoT数据采集工具")
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找生成的 exe
    exe_file = Path("dist/坦克世界AI数据采集工具.exe")
    if not exe_file.exists():
        print(f"✗ 找不到生成的 exe: {exe_file}")
        return False
    
    # 复制 exe
    shutil.copy2(exe_file, release_dir / "坦克世界AI数据采集工具.exe")
    print(f"  ✓ 复制主程序")
    
    # 复制配置文件
    config_dir = release_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(
        "wot_client/configs/client_config.yaml",
        config_dir / "client_config.yaml"
    )
    print(f"  ✓ 复制配置文件")
    
    # 创建数据目录
    (release_dir / "data" / "recordings").mkdir(parents=True, exist_ok=True)
    (release_dir / "logs").mkdir(exist_ok=True)
    print(f"  ✓ 创建数据目录")
    
    # 创建快捷启动脚本（以管理员权限运行）
    launcher_bat = release_dir / "🎮 启动（管理员权限）.bat"
    launcher_content = """@echo off
chcp 65001 > nul

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在请求管理员权限...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 运行程序
start "" "坦克世界AI数据采集工具.exe"
"""
    
    with open(launcher_bat, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    print(f"  ✓ 创建启动脚本")
    
    # 创建使用说明
    readme_content = """
═══════════════════════════════════════════════
   坦克世界 AI 数据采集工具
═══════════════════════════════════════════════

✅ 本工具无需安装 Python，可直接运行！

📝 快速开始
─────────────────────────────────────────────
  1. 双击 "🎮 启动（管理员权限）.bat"
     或者右键点击 "坦克世界AI数据采集工具.exe" 
     选择 "以管理员身份运行"
  
  2. 在配置界面中设置：
     - 屏幕分辨率（根据你的显示器）
     - 录制帧率（推荐 5 FPS）
  
  3. 点击 "🎬 开始录制" 按钮
  
  4. 进入《坦克世界》战斗
  
  5. 按 F9 键开始录制
  
  6. 正常游戏 5-10 分钟
  
  7. 按 F10 键停止录制并保存
  
  8. 可以重复步骤 5-7 录制多场战斗

⌨️ 快捷键
─────────────────────────────────────────────
  F9     → 开始录制
  F10    → 停止录制并保存
  Ctrl+C → 退出程序

📁 数据位置
─────────────────────────────────────────────
  录制的数据保存在：data/recordings/ 目录
  每场战斗单独保存在一个文件夹中
  包含：
    - frames/ 目录（游戏画面截图）
    - actions.json（键盘鼠标操作）
    - metadata.json（录制信息）

⚠️ 重要提示
─────────────────────────────────────────────
  • 必须以管理员身份运行！
    （否则无法录制键盘输入）
  
  • 建议录制分辨率：
    - 1920x1080（推荐）
    - 2560x1440
    - 3440x1440（超宽屏）
  
  • 建议帧率：5 FPS
    （平衡数据质量和磁盘空间）
  
  • 每场战斗约占用 2-5 GB 空间
  
  • 建议录制 20-50 场战斗

❓ 常见问题
─────────────────────────────────────────────
Q: 为什么键盘按键没有被录制？
A: 必须以管理员权限运行！

Q: 游戏卡顿怎么办？
A: 降低录制帧率（改为 3 FPS）或分辨率

Q: 数据保存在哪里？
A: data/recordings/ 目录下

Q: 如何修改配置？
A: 重新运行程序，在界面中修改配置

Q: 程序闪退怎么办？
A: 检查是否有杀毒软件拦截
   尝试添加到白名单

📧 问题反馈
─────────────────────────────────────────────
  如有问题，请联系开发者并提供：
  - 问题描述
  - logs/ 目录下的日志文件

═══════════════════════════════════════════════
"""
    
    with open(release_dir / "使用说明.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ 创建使用说明")
    
    # 创建快捷键说明
    hotkey_content = """
═══════════════════════════════════════════════
   快捷键说明
═══════════════════════════════════════════════

⌨️  录制快捷键
─────────────────────────────────────────────
  F9     → 开始录制
  F10    → 停止录制并保存
  Ctrl+C → 退出程序

📝 使用流程
─────────────────────────────────────────────
  1. 以管理员身份运行程序
  2. 配置分辨率和帧率
  3. 点击"开始录制"
  4. 进入游戏战斗
  5. 按 F9 开始录制
  6. 正常游戏
  7. 按 F10 停止录制
  8. 重复 5-7 录制更多战斗
  9. 按 Ctrl+C 退出

💡 提示
─────────────────────────────────────────────
  • 支持连续录制多场战斗
  • 每场战斗单独保存
  • 数据保存在 data/recordings/
  • 至少录制 5 秒才能保存
  • 建议每场录制 5-10 分钟

═══════════════════════════════════════════════
"""
    
    with open(release_dir / "快捷键说明.txt", 'w', encoding='utf-8') as f:
        f.write(hotkey_content)
    print(f"  ✓ 创建快捷键说明")
    
    print()
    print("✓ 发布包创建完成")
    
    # 压缩打包
    print("\n" + "=" * 80)
    print("📦 压缩打包")
    print("=" * 80)
    
    zip_path = Path("dist/WoT数据采集工具.zip")
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"正在压缩...")
    
    total_files = sum(1 for _ in release_dir.rglob('*') if _.is_file())
    processed = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file in release_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(release_dir.parent)
                zipf.write(file, arcname)
                processed += 1
                
                if processed % 5 == 0 or processed == total_files:
                    progress = (processed / total_files) * 100
                    print(f"  进度: {processed}/{total_files} ({progress:.1f}%)", end='\r')
    
    print()
    print()
    print("✓ 压缩完成")
    
    # 显示总结
    print("\n" + "=" * 80)
    print("✅ 打包完成！")
    print("=" * 80)
    print()
    print(f"📦 压缩包: {zip_path.absolute()}")
    print(f"📊 大小: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("=" * 80)
    print("📤 如何使用:")
    print("=" * 80)
    print()
    print("1️⃣  将 'WoT数据采集工具.zip' 发送给朋友")
    print()
    print("2️⃣  朋友解压后，双击 '🎮 启动（管理员权限）.bat'")
    print()
    print("3️⃣  或者右键 '坦克世界AI数据采集工具.exe'")
    print("    选择 '以管理员身份运行'")
    print()
    print("4️⃣  在界面中配置，然后点击 '开始录制'")
    print()
    print("5️⃣  进入游戏，按 F9 开始，F10 停止")
    print()
    print("=" * 80)
    print("✅ 非常简单！只需要一个 exe 文件！")
    print("=" * 80)
    
    return True


def main():
    """主函数"""
    print("=" * 80)
    print("🎁 坦克世界 AI - 单文件打包工具")
    print("=" * 80)
    print()
    print("本工具将程序打包成单个 exe 文件")
    print("配置和录制功能都在一个程序里")
    print("无需 Python 环境，双击即可运行")
    print()
    
    # 安装 PyInstaller
    if not install_pyinstaller():
        print("\n❌ 无法继续，请手动安装: pip install pyinstaller")
        return False
    
    # 打包
    if not build_single_exe():
        print("\n❌ 打包失败")
        return False
    
    # 创建发布包
    if not create_release_package():
        print("\n❌ 发布包创建失败")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  打包被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 打包过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

