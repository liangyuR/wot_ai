import os
from pathlib import Path
from src.core.global_context import GlobalContext
from loguru import logger

_global_context = None

def GetGlobalContext() -> GlobalContext:
    global _global_context
    if _global_context is None:
        _global_context = GlobalContext()
    return _global_context

def GetProgramDir() -> Path:
    path = Path(os.getcwd())
    return path

def MinimapBorderTemplatePath() -> str:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "minimap_border.png"

def GetHubTemplatePath() -> str:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "hub"

def GetMapTemplatePath() -> Path:
    return GetProgramDir() / "resource" / "template" / GetGlobalContext().template_tier / "map"

def GetVehicleScreenshotsDir() -> Path:
    return GetProgramDir() / "resource" / "vehicle_screenshots"

def GetMapsDir() -> Path:
    return GetProgramDir() / "resource" / "maps"

def GetConfigPath() -> Path:
    return GetProgramDir() / "config" / "config.yaml"

def GetConfigTemplatePath() -> Path:
    return GetProgramDir() / "config" / "config.yaml.template"



if __name__ == "__main__":
    logger.info(GetProgramDir())
    logger.info(MinimapBorderTemplatePath())
    logger.info(GetMapsDir())
    logger.info(GetVehicleScreenshotsDir())
    logger.info(GetConfigPath())
    logger.info(GetConfigTemplatePath())