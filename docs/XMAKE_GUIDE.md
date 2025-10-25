# xmake 构建指南

## 为什么使用 xmake？

### xmake vs CMake

| 特性 | xmake | CMake |
|------|-------|-------|
| **配置语法** | Lua（简洁） | CMake 语言（复杂）|
| **包管理** | ✅ 内置 | ❌ 需要第三方 |
| **跨平台** | ✅ 自动检测 | ⚠️ 需要手动配置 |
| **构建速度** | ⭐⭐⭐⭐⭐ 快 | ⭐⭐⭐ 中等 |
| **易用性** | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐ 复杂 |
| **社区支持** | ⭐⭐⭐ 增长中 | ⭐⭐⭐⭐⭐ 成熟 |

**优势**：
- ✅ 配置文件更简洁（50 行 vs 150 行）
- ✅ 自动处理依赖
- ✅ 更快的增量编译
- ✅ 更好的跨平台支持

---

## 安装 xmake

### Windows

**方法 1: 一键安装脚本（最简单）**
```powershell
# 右键"以管理员身份运行" PowerShell，然后执行：
.\install_xmake.ps1

# 或使用批处理脚本
.\install_xmake.bat
```

**方法 2: PowerShell 手动安装**
```powershell
# 以管理员身份运行 PowerShell
Invoke-Expression (Invoke-Webrequest 'https://xmake.io/psget.txt' -UseBasicParsing).Content
```

**方法 3: Scoop（推荐）**
```powershell
scoop install xmake
```

**方法 4: 手动下载**
1. 访问 https://github.com/xmake-io/xmake/releases
2. 下载 `xmake-vX.X.X.win64.exe`
3. 运行安装程序

### Linux

**方法 1: 一键安装（推荐）**
```bash
bash <(curl -fsSL https://xmake.io/shget.text)
```

**方法 2: 包管理器**
```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:xmake-io/xmake
sudo apt update
sudo apt install xmake

# Arch Linux
yay -S xmake

# Homebrew (macOS/Linux)
brew install xmake
```

### 验证安装
```bash
xmake --version
# 输出: xmake v2.8.5+...
```

---

## 快速开始

### 1. 配置项目

```bash
# Windows
cd d:\projects\world_of_tanks
xmake f -c -m release -p windows -a x64

# Linux
cd ~/wot_ai
xmake f -c -m release -p linux
```

**参数说明**：
- `-c`: 清理缓存
- `-m release`: Release 模式（可选 `debug`）
- `-p windows`: 平台（`windows`, `linux`, `macosx`）
- `-a x64`: 架构（`x64`, `x86`）

### 2. 构建

```bash
# 使用所有 CPU 核心构建
xmake -j

# 或使用一键构建脚本
# Windows
build_xmake.bat

# Linux
./build_xmake.sh
```

### 3. 清理

```bash
# 清理构建产物
xmake clean

# 完全清理（包括配置）
xmake clean -a
```

---

## 项目结构

### xmake.lua 配置文件

```lua
-- 项目基本信息
set_project("wot-ai")
set_version("0.1.0")
set_languages("c++17")

-- 目标：屏幕捕获库
target("screen_capture")
    set_kind("static")
    add_files("cpp/screen_capture/*.cpp")
    add_includedirs("cpp/include", {public = true})
target_end()

-- 目标：Python 绑定
target("cpp_bindings")
    set_kind("shared")
    add_files("cpp/bindings/*.cpp")
    add_deps("screen_capture", "input_control")
    add_packages("pybind11")
target_end()
```

**配置说明**：
- `set_kind()`: 目标类型（`binary`, `static`, `shared`）
- `add_files()`: 源文件
- `add_deps()`: 依赖其他目标
- `add_packages()`: 添加依赖包

---

## 常用命令

### 构建相关

```bash
# 构建所有目标
xmake

# 构建特定目标
xmake build screen_capture
xmake build cpp_bindings

# 重新构建
xmake -r

# 并行构建（使用 8 个线程）
xmake -j8

# 详细输出
xmake -v
```

### 配置相关

```bash
# 查看当前配置
xmake f --show

# 切换到 Debug 模式
xmake f -m debug

# 启用调试符号
xmake f -m debug --symbols=on

# 设置安装路径
xmake f --prefix=/usr/local
```

### 包管理

```bash
# 安装依赖包
xmake require pybind11

# 搜索包
xmake search pybind11

# 查看已安装的包
xmake require --list

# 更新包
xmake require --update
```

### 测试相关

```bash
# 构建并运行测试
xmake build test_capture
xmake run test_capture

xmake build test_input
xmake run test_input
```

### 安装

```bash
# 安装到默认路径
xmake install

# 安装到指定路径
xmake install -o python

# 卸载
xmake uninstall
```

---

## 高级用法

### 1. 条件编译

```lua
-- 根据平台添加不同的文件
if is_plat("windows") then
    add_files("cpp/windows_specific.cpp")
    add_syslinks("user32", "gdi32")
elseif is_plat("linux") then
    add_files("cpp/linux_specific.cpp")
    add_syslinks("X11")
end
```

### 2. 自定义编译选项

```lua
-- Windows MSVC
if is_plat("windows") then
    add_cxflags("/utf-8", "/W4", "/WX")
end

-- Linux GCC/Clang
if is_plat("linux") then
    add_cxflags("-Wall", "-Wextra", "-Werror")
end
```

### 3. 添加第三方库

```lua
-- 系统库
add_syslinks("pthread", "dl")

-- pkg-config 库
add_packages("opencv", "boost")

-- 自定义路径
add_includedirs("/usr/local/include")
add_linkdirs("/usr/local/lib")
add_links("mylib")
```

### 4. 交叉编译

```bash
# 为 ARM64 Linux 编译
xmake f -p linux -a arm64

# 为 Android 编译
xmake f -p android -a arm64-v8a --ndk=/path/to/ndk
```

---

## 性能优化

### 1. 增量编译

xmake 自动进行增量编译，只重新编译修改的文件：

```bash
# 第一次：编译所有文件
xmake
# 耗时: 30 秒

# 修改一个文件后
xmake
# 耗时: 2 秒 ⚡
```

### 2. 分布式编译

```bash
# 使用 distcc
xmake f --distcc=yes

# 使用 ccache
xmake f --ccache=yes
```

### 3. 预编译头

```lua
target("my_target")
    set_pcxxheader("pch.h")  -- 预编译头
    add_files("*.cpp")
```

---

## 故障排查

### 问题 1: xmake 找不到 Python

**解决**：
```bash
# 指定 Python 路径
xmake f --python=C:\Python310\python.exe

# 或设置环境变量
set PYTHON=C:\Python310\python.exe
```

### 问题 2: 找不到 pybind11

**解决**：
```bash
# 手动安装 pybind11
pip install pybind11

# 或使用 xmake 包管理
xmake require pybind11
```

### 问题 3: 编译错误

**解决**：
```bash
# 清理并重新配置
xmake clean -a
xmake f -c -m release

# 查看详细错误信息
xmake -v
```

### 问题 4: Windows 上找不到 Visual Studio

**解决**：
```bash
# 使用 xrepo 工具链
xmake f --toolchain=msvc

# 或手动指定
xmake f --vs=2022
```

---

## 与 CMake 对比

### CMake 版本（复杂）

```cmake
cmake_minimum_required(VERSION 3.15)
project(WotAI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4 /WX /O2)
else()
  add_compile_options(-Wall -Wextra -Werror -O3)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
endif()

add_library(screen_capture STATIC
  cpp/screen_capture/screen_capture.cpp
)
target_include_directories(screen_capture PUBLIC cpp/include)
if(WIN32)
  target_link_libraries(screen_capture PRIVATE user32 gdi32 d3d11 dxgi)
endif()

pybind11_add_module(cpp_bindings cpp/bindings/bindings.cpp)
target_link_libraries(cpp_bindings PRIVATE screen_capture input_control)

install(TARGETS cpp_bindings LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python)
```

**行数**: ~50 行

### xmake 版本（简洁）

```lua
set_project("wot-ai")
set_languages("c++17")
add_rules("mode.release")

add_requires("pybind11")

target("screen_capture")
    set_kind("static")
    add_files("cpp/screen_capture/*.cpp")
    add_includedirs("cpp/include", {public = true})
    if is_plat("windows") then
        add_syslinks("user32", "gdi32", "d3d11", "dxgi")
    end
target_end()

target("cpp_bindings")
    set_kind("shared")
    add_files("cpp/bindings/*.cpp")
    add_deps("screen_capture", "input_control")
    add_packages("pybind11")
    set_targetdir("$(projectdir)/python")
    set_prefixname("")
target_end()
```

**行数**: ~25 行（减少 50%）✨

---

## 迁移指南

### 从 CMake 迁移到 xmake

1. **保留 CMake**（推荐）
   ```bash
   # 两种构建系统都保留
   # 用户可以选择喜欢的方式
   
   # CMake
   build.bat
   
   # xmake
   build_xmake.bat
   ```

2. **完全迁移**
   ```bash
   # 删除 CMake 文件
   rm -rf cpp/CMakeLists.txt
   rm -rf cpp/build/
   
   # 使用 xmake
   xmake
   ```

---

## 推荐工作流

### 开发流程

```bash
# 1. 配置（只需一次）
xmake f -m debug

# 2. 开发循环
# 修改代码...
xmake          # 快速增量编译
xmake run test_capture  # 测试

# 3. 发布
xmake f -m release
xmake -r       # 重新编译
xmake install -o python
```

### 持续集成

```yaml
# .github/workflows/build.yml
- name: Setup xmake
  uses: xmake-io/github-action-setup-xmake@v1

- name: Build
  run: |
    xmake f -m release
    xmake -j
    xmake install -o python
```

---

## 总结

### xmake 优势

1. ✅ **配置简洁**: 减少 50% 配置代码
2. ✅ **自动化**: 自动检测编译器和依赖
3. ✅ **快速**: 更快的增量编译
4. ✅ **现代化**: Lua 语法，易于学习
5. ✅ **包管理**: 内置包管理器

### 何时使用

**推荐使用 xmake**：
- ✅ 新项目
- ✅ 追求简洁和效率
- ✅ 需要快速迭代
- ✅ 跨平台开发

**继续使用 CMake**：
- ⚠️ 大型遗留项目
- ⚠️ 团队熟悉 CMake
- ⚠️ 依赖 CMake 特有功能

---

## 参考资源

- 📖 [xmake 官方文档](https://xmake.io/#/zh-cn/)
- 💻 [GitHub 仓库](https://github.com/xmake-io/xmake)
- 💬 [社区讨论](https://github.com/xmake-io/xmake/discussions)
- 📺 [视频教程](https://space.bilibili.com/xxxx)

---

**推荐**: 本项目已经完全支持 xmake，使用 `build_xmake.bat` 即可开始！

