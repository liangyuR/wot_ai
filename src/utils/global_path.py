import os
import sys
from pathlib import Path
from src.utils.global_context import GlobalContext
from loguru import logger
from src.navigation.config.models import NavigationConfig
from src.navigation.config.loader import load_config

_global_context = None
_global_config = None

def GetGlobalContext() -> GlobalContext:
    global _global_context
    if _global_context is None:
        _global_context = GlobalContext()
    return _global_context

def GetGlobalConfig() -> NavigationConfig:
    global _global_config
    if _global_config is None:
        _global_config = load_config(GetConfigPath())
    return _global_config

def GetProgramDir() -> Path:
    """
    获取程序根目录路径。
    
    在打包后的环境中，返回可执行文件所在目录。
    在开发环境中，返回项目根目录。
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的环境
        # sys.executable 指向可执行文件路径
        return Path(sys.executable).parent
    else:
        # 开发环境：返回项目根目录（当前文件所在目录的父目录的父目录）
        return Path(__file__).parent.parent.parent

def MinimapBorderTemplatePath() -> str:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "minimap_border.png"

def GetHubTemplatePath() -> str:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "hub"

def GetMapTemplatePath() -> Path:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "map"

def GetMapMaskPath(map_name: str) -> Path:
    return GetProgramDir() / "resource" / "map_mask" / f"{map_name}_mask.png"

def GetVehicleScreenshotsDir() -> Path:
    return GetProgramDir() / "resource" / "vehicle_screenshots"

def GetConfigPath() -> Path:
    return GetProgramDir() / "config" / "config.yaml"

def GetLogDir() -> Path:
    return GetProgramDir() / "Logs"


if __name__ == "__main__":
    logger.info(GetProgramDir())
    logger.info(MinimapBorderTemplatePath())
    logger.info(GetMapTemplatePath())
    logger.info(GetVehicleScreenshotsDir())
    logger.info(GetConfigPath())