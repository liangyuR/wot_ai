import os
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
    path = Path(os.getcwd())
    return path

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

def GetConfigTemplatePath() -> Path:
    return GetProgramDir() / "config" / "config.yaml.template"

def GetLogDir() -> Path:
    return GetProgramDir() / "Logs"


if __name__ == "__main__":
    logger.info(GetProgramDir())
    logger.info(MinimapBorderTemplatePath())
    logger.info(GetMapTemplatePath())
    logger.info(GetVehicleScreenshotsDir())
    logger.info(GetConfigPath())
    logger.info(GetConfigTemplatePath())