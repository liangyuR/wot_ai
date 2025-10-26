"""
使用 PyInstaller 打包成独立可执行文件
不需要用户安装 Python 环境
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


def build_gui_exe():
    """打包 GUI 程序"""
    print("\n" + "=" * 80)
    print("📦 步骤 1/2: 打包 GUI 配置程序")
    print("=" * 80)
    
    cmd = [
        "pyinstaller",
        "--name=WoT数据采集配置",
        "--onefile",  # 打包成单个 exe
        "--windowed",  # 不显示控制台（GUI 程序）
        "--icon=NONE",  # 没有图标
        "--clean",
        "--distpath=dist/exe",
        "--workpath=build/gui",
        "--specpath=build",
        # 添加数据文件
        "--add-data=wot_client/configs;configs",
        "--add-data=wot_client/data_collection;data_collection",
        "--add-data=wot_client/utils;utils",
        # 隐藏导入
        "--hidden-import=win32timezone",
        "--hidden-import=win32api",
        "--hidden-import=win32con",
        "--hidden-import=pywintypes",
        # 主程序
        "wot_client/config_gui.py"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ GUI 程序打包成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ GUI 程序打包失败")
        print(f"错误输出: {e.stderr}")
        return False


def build_recorder_exe():
    """打包录制程序"""
    print("\n" + "=" * 80)
    print("📦 步骤 2/2: 打包录制程序")
    print("=" * 80)
    
    cmd = [
        "pyinstaller",
        "--name=WoT数据录制",
        "--onefile",  # 打包成单个 exe
        "--console",  # 显示控制台（方便看日志）
        "--icon=NONE",
        "--clean",
        "--distpath=dist/exe",
        "--workpath=build/recorder",
        "--specpath=build",
        # 添加数据文件
        "--add-data=wot_client/configs;configs",
        "--add-data=wot_client/data_collection;data_collection",
        "--add-data=wot_client/utils;utils",
        # 隐藏导入
        "--hidden-import=win32timezone",
        "--hidden-import=win32api",
        "--hidden-import=win32con",
        "--hidden-import=pywintypes",
        "--hidden-import=win32gui",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=mss",
        "--hidden-import=pynput",
        "--hidden-import=loguru",
        # 主程序
        "wot_client/record_main.py"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 录制程序打包成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 录制程序打包失败")
        print(f"错误输出: {e.stderr}")
        return False


def create_release_package():
    """创建最终发布包"""
    print("\n" + "=" * 80)
    print("📦 步骤 3/3: 创建发布包")
    print("=" * 80)
    
    # 创建发布目录
    release_dir = Path("dist/WoT_DataCollector_Portable")
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制可执行文件
    exe_dir = Path("dist/exe")
    if not exe_dir.exists():
        print("✗ 找不到可执行文件目录")
        return False
    
    for exe_file in exe_dir.glob("*.exe"):
        shutil.copy2(exe_file, release_dir / exe_file.name)
        print(f"  ✓ 复制 {exe_file.name}")
    
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
    
    # 创建快捷启动脚本
    quick_start = release_dir / "🎮 启动数据采集.bat"
    quick_start_content = """@echo off
chcp 65001 > nul
title 坦克世界 AI 数据采集

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo ========================================
    echo   需要管理员权限！
    echo ========================================
    echo.
    echo 请右键点击此文件，选择：
    echo   "以管理员身份运行"
    echo.
    echo 没有管理员权限将无法录制键盘输入！
    echo.
    pause
    exit /b 1
)

:: 启动配置程序
start "" "WoT数据采集配置.exe"
"""
    
    with open(quick_start, 'w', encoding='utf-8') as f:
        f.write(quick_start_content)
    print(f"  ✓ 创建启动脚本")
    
    # 创建直接录制脚本
    quick_record = release_dir / "🎬 直接开始录制.bat"
    quick_record_content = """@echo off
chcp 65001 > nul
title 坦克世界 AI - 直接录制

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo 需要管理员权限！请右键 "以管理员身份运行"
    echo.
    pause
    exit /b 1
)

echo 正在启动录制程序...
echo.

:: 启动录制程序
"WoT数据录制.exe"

pause
"""
    
    with open(quick_record, 'w', encoding='utf-8') as f:
        f.write(quick_record_content)
    print(f"  ✓ 创建直接录制脚本")
    
    # 复制说明文档
    if Path("wot_client/README_USER.md").exists():
        shutil.copy2(
            "wot_client/README_USER.md",
            release_dir / "使用说明.md"
        )
        print(f"  ✓ 复制使用说明")
    
    # 创建快捷键说明
    if Path("wot_client/快捷键说明.txt").exists():
        shutil.copy2(
            "wot_client/快捷键说明.txt",
            release_dir / "快捷键说明.txt"
        )
        print(f"  ✓ 复制快捷键说明")
    
    # 创建 README
    readme_content = """
═══════════════════════════════════════════════
   坦克世界 AI 数据采集工具 - 独立版
═══════════════════════════════════════════════

✅ 本版本无需安装 Python，可直接运行！

📁 文件说明
─────────────────────────────────────────────
  🎮 启动数据采集.bat    - 启动配置界面（推荐）
  🎬 直接开始录制.bat    - 跳过配置直接录制
  WoT数据采集配置.exe   - 配置程序
  WoT数据录制.exe       - 录制程序
  使用说明.md          - 详细使用说明
  快捷键说明.txt       - 快捷键参考

📝 快速开始
─────────────────────────────────────────────
  1. 右键点击 "🎮 启动数据采集.bat"
  2. 选择 "以管理员身份运行"
  3. 在配置界面中设置分辨率和 FPS
  4. 点击 "开始录制"
  5. 进入游戏，按 F9 开始录制

⚠️ 重要提示
─────────────────────────────────────────────
  • 必须以管理员身份运行！
  • 录制数据保存在 data/recordings/ 目录
  • 每场战斗建议录制 5-10 分钟
  • 更多详情请查看 "使用说明.md"

═══════════════════════════════════════════════
"""
    
    with open(release_dir / "README.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ 创建 README")
    
    print()
    print("✓ 发布包创建完成")
    
    # 压缩打包
    print("\n" + "=" * 80)
    print("📦 压缩打包")
    print("=" * 80)
    
    zip_path = Path("dist/WoT_DataCollector_Portable.zip")
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"正在压缩到: {zip_path.name}")
    
    total_files = sum(1 for _ in release_dir.rglob('*') if _.is_file())
    processed = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file in release_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(release_dir.parent)
                zipf.write(file, arcname)
                processed += 1
                
                if processed % 10 == 0 or processed == total_files:
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
    print(f"📁 输出目录: {release_dir.absolute()}")
    print(f"📦 压缩包: {zip_path.absolute()}")
    print(f"📊 压缩包大小: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"📄 文件总数: {total_files}")
    print()
    print("=" * 80)
    print("📤 发送给朋友:")
    print("=" * 80)
    print(f"  1. 将 {zip_path.name} 发送给朋友")
    print("  2. 解压后右键 '🎮 启动数据采集.bat'")
    print("  3. 选择 '以管理员身份运行'")
    print("  4. 无需安装 Python，可直接使用！")
    print()
    print("✅ 所有步骤完成！")
    print("=" * 80)
    
    return True


def main():
    """主函数"""
    print("=" * 80)
    print("🎁 坦克世界 AI - 独立可执行文件打包工具")
    print("=" * 80)
    print()
    print("本工具将程序打包成 .exe 文件")
    print("无需 Python 环境即可运行")
    print()
    
    # 安装 PyInstaller
    if not install_pyinstaller():
        print("\n❌ 无法继续，请手动安装 PyInstaller")
        return False
    
    # 打包 GUI
    if not build_gui_exe():
        print("\n❌ GUI 打包失败")
        return False
    
    # 打包录制程序
    if not build_recorder_exe():
        print("\n❌ 录制程序打包失败")
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

