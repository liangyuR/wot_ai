# Repository Guidelines

## Project Structure & Module Organization
- `src/` Python source: `core/` (state machine, task manager, actions), `navigation/` (path planning services/config/ui), `vision/` (detection), `attach/` (device hooks), `listeners/` (input), `ui_control/` (UI automation), `utils/global_path.py` for shared paths; entry scripts `main_window.py` and `navigation/navigation_main.py`.
- `config/` runtime settings and model paths; `resource/` static assets; `utest/` pytest cases with minimap fixtures; `.vscode/` debug configs.

## Setup, Build, and Local Run
- Install deps with `poetry install` (Python 3.11 expected). Activate the env via `poetry shell`.
- Launch main UI: `poetry run python src/main_window.py`.
- Run navigation module standalone: `poetry run python src/navigation/navigation_main.py`.
- Bundle (optional): `poetry run pyinstaller src/main_window.py` once a spec is prepared.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; order imports stdlib/third-party/local.
- Use `snake_case` for modules/functions/vars, `PascalCase` for classes, and align filenames with class purpose (e.g., `task_manager.py`).
- Prefer `loguru` over bare prints; keep settings under `config/` and shared paths via `utils.global_path`.

## Testing Guidelines
- Pytest-driven suite in `utest/`; name files/functions `test_*`.
- Run checks with `poetry run pytest utest -q`; include fixtures (e.g., minimap images) when extending coverage.
- Add regression cases for new states/tasks to keep navigation and detection stable.

## Commit & Pull Request Guidelines
- Mirror recent history: short prefix + summary (e.g., `Refactor: adjust navigation path parsing`); keep commits scoped and descriptive. If using Chinese, keep the prefix then the concise summary.
- PRs should list intent, key changes, how to reproduce/test, linked issues, and screenshots or logs for UI/vision changes.
- Call out new assets/model weights and update `config` paths when necessary.

## Security & Configuration Tips
- Keep API keys or model paths outside code; use local env vars or ignored config files.
- GPU and camera drivers (cu128, dxcam) must match the environment before running detections; document any hardware assumptions in PRs.
