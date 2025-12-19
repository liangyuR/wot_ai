# WoT AI

Automation and navigation assistant for World of Tanks. Combines state machine control, vision detection, path planning, and keyboard/mouse automation. Includes a main UI with debug view plus a navigation-only entry point.

## Requirements
- Windows, Python 3.11
- Poetry for dependency management
- GPU drivers/CUDA (e.g., cu128) and dxcam available for capture
- Game client path readable and launchable by the tool

## Setup
```bash
pip install poetry
poetry install
poetry shell
```

## Configuration
1) Create `config/config.yaml` and fill required fields: `model.path` (YOLO weights), `game.exe_path`, `minimap.template_path`.
2) Optional: set `mask.directory` or `mask.path` for custom obstacle masks; adjust `path_planning` smoothing/simplify options; pick `monitor_index` for screen capture.
3) Keep assets (templates/models) in `resource/` or any folder, and point to them via absolute or relative paths.

## Run
- Main UI with debug view:
```bash
poetry run python src/main_window.py
```
- Navigation module only:
```bash
poetry run python src/navigation/navigation_main.py
```
- Logs are written to `Logs/` for debugging navigation and detection.

## Project Layout
- `src/core/`: state machine, task manager, battle tasks, tank selector
- `src/navigation/`: path planner core, runtime services, UI, entry `navigation_main.py`
- `src/vision/`: minimap and target detection
- `src/attach/`: device hooks/interfaces
- `src/listeners/`: input listeners and hotkeys
- `src/gui/`: debug view utilities
- `src/utils/`: shared utilities and `global_path.py`
- `config/`: runtime settings
- `resource/`: static assets; `utest/`: pytest cases and minimap fixtures; `Logs/`: runtime logs

## Development and Testing
- Run tests: `poetry run pytest utest -q`
- Style: PEP 8, 4-space indent, imports ordered stdlib -> third-party -> local; prefer `loguru` over `print`.
- Before committing, run tests and sanity check that both entry points start without errors.

## Troubleshooting
- Missing model/template: verify paths in `config/config.yaml`; use absolute paths if relative ones fail.
- Permission issues: run game and this tool with sufficient privileges so capture and input hooks work.
- Performance issues: reduce `ui.overlay_fps` or lower detection confidence thresholds to lighten load.
