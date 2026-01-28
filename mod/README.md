# WoT Mod 空包

这是一个最小可用的 World of Tanks mod 空包，用于验证 mod 加载链路。

## 文件结构

```
mod/
├── res/
│   └── scripts/
│       └── client/
│           └── gui/
│               └── mods/
│                   └── mod_dc_logger.pyc  (编译后的文件)
├── mod_dc_logger.py  (源文件)
├── build.py  (构建脚本)
└── README.md  (本文件)
```

## 构建步骤

### 1. 准备 Python 2.7（推荐）

WoT 客户端使用 Python 2.7 运行 mod，因此建议使用 Python 2.7 编译脚本以确保兼容性。

**Windows:**
- 下载并安装 [Python 2.7](https://www.python.org/downloads/release/python-2718/)
- 或使用 `py -2.7`（如果已安装 Python Launcher）

**Linux/WSL:**
```bash
sudo apt-get install python2.7
```

### 2. 运行构建脚本

```bash
# Windows
python build.py

# 或使用 Python 3（会显示警告）
python3 build.py
```

构建脚本会：
1. 自动检测 Python 2.7
2. 编译 `mod_dc_logger.py` 为 `.pyc`
3. 将 `.pyc` 复制到 `res/scripts/client/gui/mods/`
4. 打包成 `dc.logger_1.0.0.wotmod`

### 3. 手动编译（可选）

如果构建脚本无法找到 Python 2.7，可以手动编译：

```bash
# Python 2.7
python2.7 -m py_compile mod_dc_logger.py

# 然后手动复制
cp mod_dc_logger.pyc res/scripts/client/gui/mods/
```

然后手动打包：

**Windows (PowerShell):**
```powershell
7z a -tzip dc.logger_1.0.0.zip res
Rename-Item dc.logger_1.0.0.zip dc.logger_1.0.0.wotmod
```

**Linux/WSL:**
```bash
zip -r dc.logger_1.0.0.zip res
mv dc.logger_1.0.0.zip dc.logger_1.0.0.wotmod
```

## 安装

1. 将生成的 `dc.logger_1.0.0.wotmod` 文件复制到游戏目录：
   ```
   <WoT游戏目录>/mods/<当前版本号>/
   ```
   例如：`C:\Games\World_of_Tanks\mods\1.23.0.0\`

2. 启动游戏

## 验证

1. 启动游戏后，打开 `python.log` 文件
   - 位置通常在：`<WoT游戏目录>/python.log`
   - 或：`<WoT游戏目录>/logs/python.log`

2. 搜索 `dc.logger`，应该能看到：
   ```
   [dc.logger] loaded!
   ```

3. 如果看到这条日志，说明 mod 加载成功 ✅

## 下一步

链路验证成功后，可以在此基础上添加功能：
- 读取游戏数据（位置、朝向等）
- 写入配置文件到 `mods/configs/<author>.<mod_id>/`
- 实现热力图数据采集等功能

## 注意事项

- `.wotmod` 文件实际上是一个 ZIP 压缩包，只是改了后缀名
- `.pyc` 文件名必须以 `mod_` 开头，否则 WoT 不会加载
- 确保使用 Python 2.7 编译以获得最佳兼容性
- 打包时保持目录结构，从 `res/` 目录开始打包

## 参考链接

- [WoT Mod Packages Documentation](https://wotmod.mtm-gaming.org/resources/packages_doc_0.4_en.pdf)
- [Wargaming Modifications Wiki](https://wiki.wargaming.net/en/Modifications)
- [Nexus Mods - WoT Mod Dev Tools](https://www.nexusmods.com/worldoftanks/mods/2480)
