# 数据格式说明

本文档描述录制数据的存储格式和结构。

## 目录结构

```
data/sessions/
└── session_20251025_154532/
    ├── meta.json           # 会话元数据
    ├── actions.json        # 操作记录（关联帧号）
    ├── frames/             # 捕获的帧（JPEG 格式）
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   ├── frame_000002.jpg
    │   └── ...
    └── gameplay.avi        # 视频（可选，仅 save_format=video 或 both）
```

## meta.json

包含会话的元数据：

```json
{
  "session_id": "session_20251025_154532",
  "timestamp": "2025-10-25 15:45:32",
  "fps": 30,
  "total_frames": 1800,
  "duration_seconds": 60.0,
  "resolution": [1920, 1080],
  "capture_mode": "fullscreen",
  "save_format": "frames",
  "frame_step": 1,
  "frames_saved": 1800
}
```

**字段说明**：
- `session_id`：会话唯一标识符
- `timestamp`：录制开始时间（人类可读格式）
- `fps`：目标帧率
- `total_frames`：总捕获帧数
- `duration_seconds`：录制时长（秒）
- `resolution`：分辨率 [宽, 高]
- `capture_mode`：捕获模式（`fullscreen` 或 `window`）
- `save_format`：保存格式（`frames`、`video` 或 `both`）
- `frame_step`：帧采样间隔（1=保存所有帧）
- `frames_saved`：实际保存的帧数

## actions.json

包含操作记录，每条记录关联到对应的帧号：

```json
[
  {
    "frame": 0,
    "timestamp": 0.0,
    "keys": ["w", "space"],
    "mouse_pos": [960, 540]
  },
  {
    "frame": 1,
    "timestamp": 0.033,
    "keys": ["w", "d"],
    "mouse_pos": [965, 542]
  },
  {
    "frame": 2,
    "timestamp": 0.067,
    "keys": ["w", "d"],
    "mouse_pos": [970, 544]
  }
]
```

**字段说明**：
- `frame`：帧号（从 0 开始）
- `timestamp`：相对时间戳（秒，从录制开始计算）
- `keys`：按下的键列表（空列表表示没有按键）
- `mouse_pos`：鼠标位置 [x, y]

**关联规则**：
- `frame` 对应 `frames/frame_{frame:06d}.jpg`
- `timestamp = frame / fps`
- 每帧都有一条操作记录

## frames/ 目录

包含捕获的游戏画面帧，JPEG 格式：

```
frames/
├── frame_000000.jpg  # 第 0 帧
├── frame_000001.jpg  # 第 1 帧
├── frame_000002.jpg  # 第 2 帧
└── ...
```

**命名规则**：
- 格式：`frame_{frame_number:06d}.jpg`
- `frame_number`：6 位数字，补零（例如：`000000`, `000001`, `000123`）
- 帧号与 `actions.json` 中的 `frame` 字段对应

**图像格式**：
- 编码：JPEG
- 质量：95%
- 色彩空间：BGR（OpenCV 默认）
- 分辨率：根据捕获源而定（通常为游戏分辨率）

**帧采样**：
- 如果 `frame_step > 1`，则只保存部分帧
- 例如 `frame_step=5` 时，只保存 `frame_000000.jpg`, `frame_000005.jpg`, `frame_000010.jpg` ...
- 但 `actions.json` 仍然包含所有帧的操作记录

## gameplay.avi（可选）

仅当 `save_format` 为 `video` 或 `both` 时生成。

**视频格式**：
- 容器：AVI
- 编解码器：XVID
- 帧率：与 `fps` 设置相同
- 分辨率：与帧图像相同

**注意**：
- `video` 模式会将所有帧存储在内存中，可能导致内存不足
- 推荐使用 `frames` 模式，后期需要视频时可通过脚本转换

## 数据加载示例

### Python

```python
import json
import cv2
from pathlib import Path

# 加载会话
session_dir = Path("data/sessions/session_20251025_154532")

# 读取元数据
with open(session_dir / "meta.json", "r") as f:
    meta = json.load(f)
print(f"FPS: {meta['fps']}, 总帧数: {meta['total_frames']}")

# 读取操作记录
with open(session_dir / "actions.json", "r") as f:
    actions = json.load(f)

# 加载特定帧
frame_0 = cv2.imread(str(session_dir / "frames" / "frame_000000.jpg"))

# 遍历所有帧和操作
for action in actions:
    frame_num = action["frame"]
    frame_path = session_dir / "frames" / f"frame_{frame_num:06d}.jpg"
    
    # 检查帧是否存在（考虑 frame_step）
    if frame_path.exists():
        frame = cv2.imread(str(frame_path))
        keys = action["keys"]
        mouse = action["mouse_pos"]
        
        # 处理数据...
        print(f"帧 {frame_num}: 按键={keys}, 鼠标={mouse}")
```

## 最佳实践

### 存储优化

1. **使用 `frame_step`**：
   - 如果不需要高频数据，使用 `--frame-step 5` 可节省 80% 存储
   - 推荐值：
     - 训练模仿学习：`frame_step=1` 或 `frame_step=2`
     - 数据分析：`frame_step=5` 或 `frame_step=10`
     - 演示视频：`frame_step=1`

2. **使用 `frames` 模式**：
   - 实时保存到磁盘，内存占用小
   - 支持长时间录制（小时级别）
   - 可选择性加载帧，无需全部加载到内存

3. **JPEG 质量设置**：
   - 当前固定为 95%，提供高质量和合理的文件大小
   - 1920x1080 帧约 200-300 KB
   - 30 FPS 1 分钟约 360-540 MB

### 数据完整性

1. **检查 `total_frames` 与实际帧数**：
   ```python
   expected_frames = meta["frames_saved"]
   actual_frames = len(list((session_dir / "frames").glob("*.jpg")))
   if expected_frames != actual_frames:
       print("警告：帧数不匹配！")
   ```

2. **验证 actions 与 frames 对齐**：
   ```python
   for action in actions:
       frame_path = session_dir / "frames" / f"frame_{action['frame']:06d}.jpg"
       if not frame_path.exists() and action['frame'] % meta['frame_step'] == 0:
           print(f"警告：缺失帧 {action['frame']}")
   ```

## 版本历史

### v2.0（当前版本，2025-10-25）
- ✨ 引入 session 目录结构
- ✨ 添加 `meta.json` 元数据
- ✨ `actions.json` 使用帧号关联
- ✨ 实时保存 JPEG 帧
- ✨ 支持帧采样（`frame_step`）

### v1.0（旧版本）
- 保存为单个视频文件（`gameplay.mp4`）
- `actions.json` 使用时间戳
- 所有帧存储在内存中

## 迁移指南

如果你有旧版本的数据（v1.0），可以使用以下脚本转换：

```python
# TODO: 提供转换脚本
```

## 参考项目

本数据格式借鉴了以下项目的最佳实践：
- [record-gameplay](https://github.com/Oughie/record-gameplay)（帧保存、session 结构）

## 常见问题

### Q: 为什么不直接保存视频？
A: 
- 视频需要将所有帧存储在内存中，长时间录制会导致内存溢出
- 帧模式支持实时保存，无内存限制
- 后期处理更灵活（可选择性加载、数据增强等）

### Q: 如何将帧转换为视频？
A:
```bash
# 使用 ffmpeg
cd data/sessions/session_20251025_154532/frames
ffmpeg -framerate 30 -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p ../output.mp4
```

### Q: frame_step 会影响 actions.json 吗？
A: 不会。`actions.json` 始终包含所有帧的操作记录。`frame_step` 只影响保存的图像帧数量。

