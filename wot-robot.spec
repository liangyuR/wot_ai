# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for wot-robot
"""

import os
from pathlib import Path

block_cipher = None

# 获取项目根目录
project_root = Path(SPECPATH)

a = Analysis(
    ['src/main_window.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config', 'config'),  # 配置文件目录
        ('resource', 'resource'),  # 资源文件目录
    ],
    hiddenimports=[
        'src.core.battle_task',
        'src.core.state_machine',
        'src.core.tank_selector',
        'src.navigation',
        'src.navigation.config',
        'src.navigation.config.loader',
        'src.navigation.config.models',
        'src.navigation.controller',
        'src.navigation.nav_runtime',
        'src.navigation.path_planner',
        'src.navigation.service',
        'src.vision',
        'src.utils',
        'loguru',
        'cv2',
        'PIL',
        'numpy',
        'ultralytics',
        'pydantic',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
        'test',
        'tests',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='wot-robot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI应用，不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标文件路径，例如: 'resource/icon.ico'
)

