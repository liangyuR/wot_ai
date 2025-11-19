import os
from pathlib import Path
from src.core.global_context import GlobalContext
from loguru import logger

def GetProgramDir() -> Path:
    path = Path(os.getcwd())
    return path

def MinimapBorderTemplatePath() -> str:
    return GetProgramDir() / "resource" / "template" / GlobalContext().template_tier / "minimap_border.png"

def TemplatePath(template_name: str) -> str:
    return GetProgramDir() / "resource" / "template" / GlobalContext().template_tier / template_name

def GetMapsDir() -> Path:
    return GetProgramDir() / "resource" / "maps"

def GetVehicleScreenshotsDir() -> Path:
    return GetProgramDir() / "resource" / "vehicle_screenshots"

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