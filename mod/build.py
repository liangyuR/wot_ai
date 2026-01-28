#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WoT Mod 构建脚本
编译 Python 2.7 脚本并打包成 .wotmod 文件
"""
import os
import sys
import shutil
import zipfile
import subprocess
import py_compile

MOD_NAME = 'dc.logger'
MOD_VERSION = '1.0.0'
SOURCE_FILE = 'mod_dc_logger.py'
TARGET_DIR = 'res/scripts/client/gui/mods'
OUTPUT_FILE = '{}_{}.wotmod'.format(MOD_NAME, MOD_VERSION)


def find_python27():
    """尝试找到 Python 2.7 可执行文件"""
    possible_paths = [
        'python2.7',
        'python2',
        'py -2.7',
        'py -2',
    ]
    
    # Windows 特定路径
    if sys.platform == 'win32':
        possible_paths.extend([
            r'C:\Python27\python.exe',
            r'C:\Python27\pythonw.exe',
        ])
    
    for cmd in possible_paths:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if '2.7' in result.stdout or '2.7' in result.stderr:
                return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    return None


def compile_with_python27(python_cmd, source_file, output_file):
    """使用 Python 2.7 编译脚本"""
    try:
        # 使用 py_compile 模块编译
        subprocess.run(
            [python_cmd, '-m', 'py_compile', source_file],
            check=True,
            cwd=os.path.dirname(os.path.abspath(source_file)) or '.'
        )
        print(f'✓ 使用 {python_cmd} 编译成功')
        return True
    except subprocess.CalledProcessError as e:
        print(f'✗ 编译失败: {e}')
        return False
    except Exception as e:
        print(f'✗ 编译出错: {e}')
        return False


def compile_with_current_python(source_file):
    """使用当前 Python 版本编译（可能不兼容，但可以尝试）"""
    try:
        py_compile.compile(source_file, doraise=True)
        print('✓ 使用当前 Python 版本编译成功（警告：可能不兼容 Python 2.7）')
        return True
    except py_compile.PyCompileError as e:
        print(f'✗ 编译失败: {e}')
        return False
    except Exception as e:
        print(f'✗ 编译出错: {e}')
        return False


def build_mod():
    """构建 mod 包"""
    print('=' * 60)
    print('WoT Mod 构建脚本')
    print('=' * 60)
    
    # 检查源文件
    if not os.path.exists(SOURCE_FILE):
        print(f'✗ 错误: 找不到源文件 {SOURCE_FILE}')
        return False
    
    # 创建目标目录
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f'✓ 创建目录: {TARGET_DIR}')
    
    # 尝试找到 Python 2.7
    python27 = find_python27()
    
    # 编译源文件
    compiled_file = SOURCE_FILE + 'c'
    if python27:
        print(f'找到 Python 2.7: {python27}')
        if not compile_with_python27(python27, SOURCE_FILE, compiled_file):
            print('⚠ 警告: Python 2.7 编译失败，尝试使用当前 Python 版本')
            if not compile_with_current_python(SOURCE_FILE):
                return False
    else:
        print('⚠ 警告: 未找到 Python 2.7，使用当前 Python 版本编译')
        print('   注意: 生成的 .pyc 可能不兼容 WoT 的 Python 2.7 环境')
        if not compile_with_current_python(SOURCE_FILE):
            return False
    
    # 检查编译后的文件
    if not os.path.exists(compiled_file):
        print(f'✗ 错误: 编译后的文件不存在: {compiled_file}')
        return False
    
    # 复制到目标目录
    target_file = os.path.join(TARGET_DIR, os.path.basename(compiled_file))
    shutil.copy2(compiled_file, target_file)
    print(f'✓ 复制到: {target_file}')
    
    # 清理临时文件
    if os.path.exists(compiled_file):
        os.remove(compiled_file)
        print(f'✓ 清理临时文件: {compiled_file}')
    
    # 打包成 .wotmod
    print(f'\n开始打包: {OUTPUT_FILE}')
    try:
        with zipfile.ZipFile(OUTPUT_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 从 res/ 目录开始打包
            for root, dirs, files in os.walk('res'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = file_path.replace(os.sep, '/')
                    zipf.write(file_path, arc_name)
                    print(f'  + {arc_name}')
        
        print(f'\n✓ 打包成功: {OUTPUT_FILE}')
        print(f'  文件大小: {os.path.getsize(OUTPUT_FILE)} 字节')
        return True
    except Exception as e:
        print(f'✗ 打包失败: {e}')
        return False


if __name__ == '__main__':
    success = build_mod()
    sys.exit(0 if success else 1)
