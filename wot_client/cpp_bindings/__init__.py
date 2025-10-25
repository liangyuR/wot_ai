"""
WoT AI C++ Bindings
High-performance screen capture and input control modules
"""

try:
    from .cpp_bindings import (
        ScreenCapture,
        InputControl,
        MouseButton
    )
    __all__ = ['ScreenCapture', 'InputControl', 'MouseButton']
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import C++ bindings: {e}\n"
        "C++ modules are not available. The system will fall back to Python implementations.\n"
        "To enable C++ acceleration, run: build_xmake.bat",
        ImportWarning
    )
    
    # 提供空类以避免导入错误（仅在导入失败时）
    ScreenCapture = None
    InputControl = None
    MouseButton = None
    
    __all__ = []

