# dc_mod (player position logger)

Minimal World of Tanks mod that periodically logs the local player's vehicle
position to `python.log`.

## Files

```
dc_mod/
  mod_dc_position.py
  build.py
  res/scripts/client/gui/mods/   (created by build)
```

## Build

Recommended: Python 2.7 (WoT client runtime).

```powershell
cd D:\project\wot_ai\dc_mod
python build.py
```

This produces:
```
dc.position_1.0.0.wotmod
```

## Install

Copy the `.wotmod` to:
```
<WoT>/mods/<current_version>/
```

Example:
```
C:\Games\World_of_Tanks\mods\1.25.0.0\
```

## Verify

Open the game's `python.log` and search for:
```
[dc.position] loaded
[dc.position] pos=...
```

## Notes

- Logging happens every second and only when the vehicle moved at least 0.5 m.
- This mod only reads the local player's position and prints to `python.log`.
- On load, the mod also posts a client message in the bottom-right chat/message
  window: "dc.position loaded".
