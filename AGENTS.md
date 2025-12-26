# Repository Guidelines

## Project Structure & Module Organization
- Source under `src/`: `core/` (state machine, task manager, actions), `navigation/` (path planning services/config/ui), `vision/` (detection), `attach/` (device hooks), `listeners/` (input), `ui_control/` (UI automation).
- Config and assets in `config/` (runtime settings, model paths) and `resource/` (static files); shared paths via `src/utils/global_path.py`.
- Entry points: `src/main_window.py` (main UI) and `src/navigation/navigation_main.py` (navigation-only); tests live in `utest/` with minimap fixtures.

## Build, Test, and Development Commands
- Install deps (Python 3.11): `poetry install` then `poetry shell` to activate the venv.
- Run main UI: `poetry run python src/main_window.py`.
- Run navigation module standalone: `poetry run python src/navigation/navigation_main.py`.
- Execute tests: `poetry run pytest utest -q` (quiet run of the suite).
- Optional bundle: `poetry run pyinstaller src/main_window.py` once a spec exists.

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indents; imports ordered stdlib -> third-party -> local.
- Naming: snake_case for modules/functions/vars, PascalCase for classes; align filenames with class purpose (e.g., `task_manager.py`).
- Prefer `loguru` over bare prints; keep settings in `config/` and shared paths in `utils/global_path`.

## Testing Guidelines
- Framework: pytest with tests in `utest/`; name files and functions `test_*`.
- Run `poetry run pytest utest -q` before changes land; add regression cases for new states/tasks in navigation/detection.
- Include fixtures (e.g., minimap images) when extending coverage to keep navigation stable.

## Commit & Pull Request Guidelines
- Commits: short prefix + summary (e.g., `Refactor: adjust navigation path parsing`); keep scope focused.
- PRs: include intent, key changes, reproduction steps/commands, linked issues, and screenshots/logs for UI or vision updates; call out new assets/model weights and update `config` paths when needed.

## Security & Configuration Tips
- Do not hardcode API keys or model paths; use env vars or ignored config files.
- Confirm GPU/camera drivers (cu128, dxcam) match environment before running detections; document hardware assumptions in PRs.
