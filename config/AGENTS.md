# AGENTS

- Purpose: quick reference for adjusting `config/config.yaml` without breaking navigation.
- Encoding: UTF-8 (ASCII chars only) to avoid garbled text.

## Path Planning Quick Tips
- `enable_astar_smoothing`: keep `true` for smoother A* output.
- `astar_smooth_weight`: 0.4-0.6 balances smoothness vs. detours; larger is smoother but may hug obstacles; smaller is straighter but jerkier.
- `post_smoothing_method`: `catmull_rom` for curves; `los` for straight-line simplification.
- `num_points_per_segment`: higher = finer curve, more points; lower = coarser.
- `simplify_threshold`: lower (3-6) keeps safety margins near 12-15px paths; higher (>8) shortens paths but risks sharp turns/obstacle proximity.
- `curvature_threshold_deg`: 25-35 makes turns gentler for a ~3px agent; larger tolerates sharper turns.

## Safety & Paths
- Keep model/template paths valid: `model.path`, `minimap.template_path`, `game.exe_path`.
- Use absolute paths if relative paths fail; avoid hardcoding secrets.
- If capture/input fails, run with sufficient privileges.

## Testing
- Run `poetry run pytest utest -q` after config-dependent changes when possible.
- Smoke test: `poetry run python src/navigation/navigation_main.py` to verify navigation and overlays start.
