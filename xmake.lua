-- World of Tanks AI C++ 项目配置
-- 使用 xmake 进行构建管理

-- 设置项目信息
set_project("wot-ai")
set_version("0.1.0")
set_languages("c++17")

-- 设置构建模式
add_rules("mode.debug", "mode.release")

-- Google C++ 代码规范
set_warnings("all", "error")

-- 优化选项
if is_mode("release") then
    set_optimize("fastest")
    set_strip("all")
end

-- Windows 平台配置
if is_plat("windows") then
    add_cxflags("/utf-8")
    -- 添加 Windows SDK
    add_syslinks("user32", "gdi32", "d3d11", "dxgi")
end

-- 添加包依赖
add_requires("pybind11")

-- 屏幕捕获静态库
target("screen_capture")
    set_kind("static")
    
    -- Linux 需要 PIC（Position Independent Code）
    if is_plat("linux") then
        add_cxflags("-fPIC")
    end
    
    -- 根据平台选择源文件
    if is_plat("windows") then
        add_files("wot_client/cpp/screen_capture/screen_capture.cpp")
        add_syslinks("user32", "gdi32", "d3d11", "dxgi")
    elseif is_plat("linux") then
        add_files("wot_client/cpp/screen_capture/screen_capture_linux.cpp")
        -- X11 是可选的，检测后自动链接
        add_packages("libx11", {optional = true})
        add_syslinks("X11", "Xext", {optional = true})
    end
    
    add_includedirs("wot_client/cpp/include", {public = true})
target_end()

-- 输入控制静态库
target("input_control")
    set_kind("static")
    
    -- Linux 需要 PIC
    if is_plat("linux") then
        add_cxflags("-fPIC")
    end
    
    add_files("wot_client/cpp/input_control/*.cpp")
    add_includedirs("wot_client/cpp/include", {public = true})
    
    if is_plat("windows") then
        add_syslinks("user32")
    elseif is_plat("linux") then
        add_syslinks("X11", "Xtst")
    end
target_end()

-- Python 绑定模块
target("cpp_bindings")
    set_kind("shared")
    add_files("wot_client/cpp/bindings/*.cpp")
    add_deps("screen_capture", "input_control")
    add_packages("pybind11")
    
    -- Python 扩展模块配置
    set_extension(".pyd")  -- Windows
    if is_plat("linux") then
        set_extension(".so")
    end
    
    -- 输出到客户端 cpp_bindings 目录
    set_targetdir("$(projectdir)/wot_client/cpp_bindings")
    
    -- 移除 lib 前缀
    set_prefixname("")
target_end()

-- 测试目标（可选）
target("test_capture")
    set_kind("binary")
    add_files("wot_client/cpp/tests/test_capture.cpp")
    add_deps("screen_capture")
    set_default(false)  -- 默认不编译
target_end()

target("test_input")
    set_kind("binary")
    add_files("wot_client/cpp/tests/test_input.cpp")
    add_deps("input_control")
    set_default(false)  -- 默认不编译
target_end()

