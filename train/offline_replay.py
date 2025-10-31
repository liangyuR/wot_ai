#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline replay evaluator: overlay model predictions and ground-truth on frames,
then export to a video (mp4) or a folder of annotated frames.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # fallback to imageio

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:
    imageio = None

from train_imitation import TankImitationNet, TankImitationDataset


SCREEN_W, SCREEN_H = 3840, 2160
ACTION_KEYS = ["w", "a", "s", "d"]
MOUSE_KEYS = ["mouse_left", "mouse_right"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_prediction(pred: np.ndarray) -> Dict[str, Any]:
    key_logits = pred[:6]
    probs = sigmoid(key_logits)
    keys_pressed = []
    names = ACTION_KEYS + MOUSE_KEYS
    for k, p in zip(names, probs):
        if p > 0.5:
            keys_pressed.append(k)
    mx = float(pred[6]) * SCREEN_W
    my = float(pred[7]) * SCREEN_H
    return {
        "keys": keys_pressed,
        "mouse": (mx, my),
        "probs": dict(zip(names, probs.tolist())),
    }


def decode_truth(truth: Dict[str, Any]) -> Dict[str, Any]:
    keys = truth.get("keys", [])
    mx, my = truth.get("mouse_pos", [SCREEN_W // 2, SCREEN_H // 2])
    return {"keys": keys, "mouse": (float(mx), float(my))}


def draw_overlay_pil(img: Image.Image, pred: Dict[str, Any], truth: Dict[str, Any]) -> Image.Image:
    draw = ImageDraw.Draw(img)

    # Colors
    green = (0, 255, 0)
    red = (255, 0, 0)
    yellow = (255, 255, 0)

    # Mice positions
    r = 10
    px, py = pred["mouse"]
    tx, ty = truth["mouse"]
    # predicted mouse (green)
    draw.ellipse([px - r, py - r, px + r, py + r], outline=green, width=3)
    # ground-truth mouse (red)
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline=red, width=3)

    # Keys text
    text_lines = [
        f"PRED keys: {'+'.join(pred['keys']) if pred['keys'] else '-'}",
        f"TRUE keys: {'+'.join(truth['keys']) if truth['keys'] else '-'}",
        f"PRED mouse: ({int(px)}, {int(py)})  TRUE mouse: ({int(tx)}, {int(ty)})",
    ]
    y = 10
    for line in text_lines:
        draw.rectangle([10, y, 10 + 1000, y + 28], fill=(0, 0, 0, 120))
        draw.text((16, y + 6), line, fill=yellow)
        y += 34

    return img


def annotate_frame(img_path: Path, pred: Dict[str, Any], truth: Dict[str, Any]) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = draw_overlay_pil(img, pred, truth)
    frame = np.array(img)[:, :, ::-1] if cv2 is not None else np.array(img)
    return frame


def export_video(frames: list, output_path: Path, fps: int = 5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for f in frames:
            # Ensure BGR for OpenCV
            if f.shape[2] == 3 and f[..., ::-1].copy() is not None:
                # assume already BGR from annotate_frame
                pass
            vw.write(f)
        vw.release()
    elif imageio is not None:
        imageio.mimsave(str(output_path), frames, fps=fps)
    else:
        # Fallback: dump frames as images
        dump_dir = output_path.with_suffix("")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f[:, :, ::-1] if cv2 is not None else f).save(dump_dir / f"frame_{i:04d}.png")


def main():
    parser = argparse.ArgumentParser(description="Offline replay evaluator")
    parser.add_argument("--model", type=str, default=str(Path(__file__).parent / "tank_imitation_model.pth"))
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent.parent / "data_collection" / "data" / "recordings"))
    parser.add_argument("--session", type=str, default="", help="optional session id filter, e.g. session_20251030_010955")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output", type=str, default=str(Path("replay.mp4")))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TankImitationNet(action_dim=8).to(device).eval()
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Dataset (we only use it to build sample index and transforms)
    ds = TankImitationDataset(args.data_root)

    frames = []
    count = 0
    for idx, (frame_path, action) in enumerate(ds.samples):
        if args.session and (Path(frame_path).parents[1].name != args.session):
            continue
        # Load original frame path
        img_path = Path(frame_path)

        # Preprocess like training
        img_input = ds.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out = model(img_input)
            else:
                out = model(img_input)
        pred = out[0].detach().cpu().numpy()
        pred_decoded = decode_prediction(pred)
        truth_decoded = decode_truth(action)

        # Annotate
        frame = annotate_frame(img_path, pred_decoded, truth_decoded)
        frames.append(frame)

        count += 1
        if count >= args.max_frames:
            break

    if not frames:
        print("No frames collected. Check filters and data_root.")
        return

    export_video(frames, Path(args.output), fps=args.fps)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()


