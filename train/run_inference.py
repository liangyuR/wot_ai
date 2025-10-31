#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal in-battle controller: load imitation model, capture screen, predict actions,
and send keyboard/mouse to control the tank during battle.

Hotkeys:
  - F9: start control
  - F10: stop control
  - ESC: emergency stop & exit

Notes:
  - Manually select tank / enter battle; this tool only controls in-battle.
  - Default backend: pynput (simple). For fullscreen issues, consider Win32 later.
"""

import argparse
import time
import math
import threading
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import mss
from pynput import keyboard as kb_listener
from pynput import keyboard
from pynput import mouse

# Local import of model definition
# Robust import of model definition when running as a script
try:
    from train.train_imitation import TankImitationNet, TARGET_W, TARGET_H  # when project root is on sys.path
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from train_imitation import TankImitationNet, TARGET_W, TARGET_H  # local sibling import


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PynputController:
    """Keyboard/Mouse output using pynput."""
    def __init__(self):
        self.kb = keyboard.Controller()
        self.ms = mouse.Controller()
        self.pressed = set()

    def press_key(self, key_char: str):
        if key_char in self.pressed:
            return
        self.kb.press(key_char)
        self.pressed.add(key_char)

    def release_key(self, key_char: str):
        if key_char not in self.pressed:
            return
        self.kb.release(key_char)
        self.pressed.discard(key_char)

    def click(self, button: str, pressed: bool):
        btn = mouse.Button.left if button == 'left' else mouse.Button.right
        if pressed:
            self.ms.press(btn)
        else:
            self.ms.release(btn)

    def move_abs(self, x: int, y: int):
        self.ms.position = (int(x), int(y))

    def move_rel(self, dx: int, dy: int):
        self.ms.move(int(dx), int(dy))

    def release_all(self):
        for k in list(self.pressed):
            try:
                self.kb.release(k)
            except Exception:
                pass
        self.pressed.clear()


def build_transform(width: int, height: int):
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ActionPostProcessor:
    """Hysteresis for keys + mouse smoothing and max step limiting."""
    def __init__(self, screen_w: int, screen_h: int, key_on: float, key_off: float,
                 mouse_alpha: float, max_move_px: int, abs_mouse: bool):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.key_on = key_on
        self.key_off = key_off
        self.mouse_alpha = mouse_alpha
        self.max_move_px = max_move_px
        self.abs_mouse = abs_mouse

        self.key_names = ['w', 'a', 's', 'd', 'mouse_left', 'mouse_right']
        self.key_state = {k: False for k in self.key_names}
        self.smoothed_pos = None  # (x, y)

    def update_keys(self, key_probs: np.ndarray) -> dict:
        """Apply hysteresis on 6 key probabilities, return dict of desired states."""
        desired = {}
        for i, name in enumerate(self.key_names):
            p = float(key_probs[i])
            cur = self.key_state[name]
            if cur:
                nxt = p >= self.key_off  # stay on until drops below off
            else:
                nxt = p >= self.key_on   # turn on when above on
            desired[name] = nxt
        self.key_state.update(desired)
        return desired

    def update_mouse(self, xy_norm: np.ndarray, current_pos: tuple) -> tuple:
        """Return target absolute pos with smoothing and max step limiting (absolute mode)."""
        tx = float(np.clip(xy_norm[0], 0.0, 1.0)) * self.screen_w
        ty = float(np.clip(xy_norm[1], 0.0, 1.0)) * self.screen_h
        target = (tx, ty)

        if self.smoothed_pos is None:
            self.smoothed_pos = target
        else:
            ax = self.mouse_alpha
            self.smoothed_pos = (ax * target[0] + (1 - ax) * self.smoothed_pos[0],
                                 ax * target[1] + (1 - ax) * self.smoothed_pos[1])

        sx, sy = self.smoothed_pos
        cx, cy = current_pos
        dx = sx - cx
        dy = sy - cy
        dist = math.hypot(dx, dy)
        if dist > self.max_move_px:
            scale = self.max_move_px / (dist + 1e-6)
            sx = cx + dx * scale
            sy = cy + dy * scale
        return (int(sx), int(sy))


def capture_frame(sct: mss.mss, monitor: dict) -> Image.Image:
    img_bgra = np.array(sct.grab(monitor))  # H, W, 4
    img_rgb = img_bgra[:, :, :3][:, :, ::-1]  # to RGB
    return Image.fromarray(img_rgb)


def infer_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = TankImitationNet(action_dim=8).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Screen capture init
    sct = mss.mss()
    mon = sct.monitors[1]
    screen_w = mon['width']
    screen_h = mon['height']
    monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
    logger.info(f"Screen: {screen_w}x{screen_h}")

    tfm = build_transform(TARGET_W, TARGET_H)
    controller = PynputController()
    post = ActionPostProcessor(
        screen_w=screen_w,
        screen_h=screen_h,
        key_on=args.key_th_on,
        key_off=args.key_th_off,
        mouse_alpha=args.mouse_alpha,
        max_move_px=args.max_move_px,
        abs_mouse=args.abs_mouse,
    )

    # Hotkey state
    enabled_flag = {'on': False}
    stop_flag = {'stop': False}

    def on_press(key):
        try:
            if key == kb_listener.Key.f9:
                enabled_flag['on'] = True
                logger.info("Control: ON (F9)")
            elif key == kb_listener.Key.f10:
                enabled_flag['on'] = False
                controller.release_all()
                logger.info("Control: OFF (F10)")
            elif key == kb_listener.Key.esc:
                enabled_flag['on'] = False
                controller.release_all()
                stop_flag['stop'] = True
                logger.info("Emergency stop (ESC)")
        except Exception:
            pass

    listener = kb_listener.Listener(on_press=on_press)
    listener.start()

    # Main loop
    interval = 1.0 / float(args.fps)
    last_log = time.time()
    frames = 0

    logger.info("Switch to the game window. Press F9 to start, F10 to stop, ESC to exit.")
    try:
        while not stop_flag['stop']:
            t0 = time.time()

            # Capture and preprocess
            frame = capture_frame(sct, monitor)
            x = tfm(frame).unsqueeze(0).to(device, non_blocking=True)

            # Inference (AMP)
            with torch.no_grad():
                amp_ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
                with amp_ctx:
                    out = model(x)  # (1, 8)
            out_np = out.squeeze(0).detach().cpu().numpy()

            # Post-process
            if enabled_flag['on']:
                key_logits = out_np[:6]
                key_probs = 1.0 / (1.0 + np.exp(-key_logits))
                desired = post.update_keys(key_probs)

                # Apply keys
                for name, on in desired.items():
                    if name == 'mouse_left' or name == 'mouse_right':
                        # Mouse buttons
                        btn = 'left' if name == 'mouse_left' else 'right'
                        controller.click(btn, on)
                    else:
                        # Keyboard WASD
                        (controller.press_key(name) if on else controller.release_key(name))

                # Mouse control
                mouse_pos = controller.ms.position
                if args.abs_mouse:
                    # Interpret outputs as absolute normalized coords
                    target_pos = post.update_mouse(out_np[6:], mouse_pos)
                    controller.move_abs(*target_pos)
                else:
                    # Interpret outputs as relative movement (dx, dy) in [-1,1]
                    raw_dx, raw_dy = out_np[6], out_np[7]
                    # tanh to bound
                    dx = math.tanh(float(raw_dx)) * args.max_move_px
                    dy = math.tanh(float(raw_dy)) * args.max_move_px
                    # optional simple EMA smoothing on deltas
                    if post.smoothed_pos is None:
                        sdx, sdy = dx, dy
                    else:
                        # reuse mouse_alpha as delta smoothing factor
                        a = post.mouse_alpha
                        sdx = a * dx
                        sdy = a * dy
                    controller.move_rel(int(sdx), int(sdy))
            else:
                # Ensure no sticky keys when disabled
                controller.release_all()

            # pacing
            frames += 1
            now = time.time()
            if now - last_log >= 2.0:
                logger.info(f"Loop FPS ~ {frames / (now - last_log):.1f} | enabled={enabled_flag['on']}")
                last_log = now
                frames = 0

            spent = time.time() - t0
            sleep_t = max(0.0, interval - spent)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        enabled_flag['on'] = False
        controller.release_all()
        listener.stop()
        logger.info("Controller stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Run in-battle model controller")
    p.add_argument('--model', type=str, default=str(Path(__file__).parent / 'tank_imitation_model.pth'))
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--key-th-on', type=float, default=0.55, dest='key_th_on')
    p.add_argument('--key-th-off', type=float, default=0.45, dest='key_th_off')
    p.add_argument('--mouse-alpha', type=float, default=0.4)
    p.add_argument('--max-move-px', type=int, default=40)
    p.add_argument('--abs-mouse', action='store_true', default=False)
    p.add_argument('--rel-mouse', dest='abs_mouse', action='store_false')
    return p.parse_args()


def main():
    args = parse_args()
    infer_loop(args)


if __name__ == '__main__':
    main()


