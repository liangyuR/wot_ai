# Attack Module 设计文档（attack-module.md）

> 目标：在现有导航与小地图体系之上，实现一个可扩展的攻击模块，用于自动完成“发现目标 → 瞄准 → 开火”的流程，支持后续行为克隆与策略升级。

---

## 1. 总体架构概览

攻击模块分为 4 层，由下到上分别是：

1. **感知层（Perception Layer）**

   * 负责在主视野中发现敌方目标（车身 + 血条/名字条）。
   * 输出屏幕坐标系下的目标列表 `ScreenTarget`。

2. **目标选择层（Target Selection Layer）**

   * 根据目标距离、屏幕位置、可见性等信息，选择当前要攻击的“主目标”。

3. **瞄准控制层（Aim Control Layer）**

   * 将“主目标”的屏幕坐标 → 转换为鼠标移动/炮塔调整命令，逐帧逼近目标。

4. **射击决策层（Fire Control Layer）**

   * 监控瞄准状态（准心是否对准 + 缩圈是否完成）与装填状态，决定何时执行左键开火。

攻击模块与现有系统的关系：

* 输入：

  * 主视野截图（来自 CaptureService / 截图线程）。
  * 当前游戏状态（是否在战斗中、被击毁、装填状态等，初版可部分硬编码）。
* 输出：

  * 键鼠控制命令（移动鼠标、按下/释放鼠标左键，未来可扩展键盘输入）。

攻击模块本身应当是**无状态或弱状态**的服务，由上层 Runtime 驱动其 `update()` 接口（例如固定 20 FPS）。

---

## 2. 核心数据结构

### 2.1 ScreenTarget

代表主视野中一个可攻击目标（敌方坦克）：

```python
@dataclass
class ScreenTarget:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in screen px
    center: Tuple[int, int]          # (cx, cy)
    score: float                     # detection confidence
    cls_id: int                      # 类别 id（主视野车身类别）
    is_enemy: bool                   # 是否为敌方
    est_distance: Optional[float] = None  # 粗略“距离”指标（例如 1 / bbox_height）
```

### 2.2 AimCommand

瞄准控制层内部使用的“期望鼠标运动”命令：

```python
@dataclass
class AimCommand:
    dx: int      # 鼠标在 x 方向移动
    dy: int      # 鼠标在 y 方向移动
    is_stable: bool  # 准星是否已基本稳定在目标附近
```

### 2.3 FireDecision

射击决策输出：

```python
@dataclass
class FireDecision:
    should_fire: bool  # 本帧是否执行一次左键点击
    reason: str        # 触发原因，方便调试，比如 "aim_ok_and_circle_small"
```

---

## 3. 模块拆分设计

### 3.1 主视野检测器（MainViewDetector）

**职责：**

* 使用两个 YOLO DET 模型检测：

  * 模型 A：车身 `tank_body`
  * 模型 B：敌方血条/名字条 `enemy_ui`
* 做血条 → 车身的关联，过滤出“敌方且可攻击的坦克”。

**核心接口：**

```python
class MainViewDetector:
    def __init__(
        self,
        tank_detector,    # YOLO 模型实例（车身）
        hpbar_detector,   # YOLO 模型实例（血条/红名）
        conf_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        ...

    def detect(self, frame_bgr: np.ndarray) -> List[ScreenTarget]:
        """在主视野 frame 中检测所有可攻击目标。"""
        ...
```

**备注：**

* 只关注第三人称视角画面（不开镜）。
* 输入分辨率统一为 1280×720。
* 宽松 BBOX 标注车身：看到一点就框整车。

---

### 3.2 目标选择器（AttackTargetSelector）

**职责：**

* 从 `ScreenTarget` 列表中选出当前帧的主攻击目标。
* 目标切换时加入一定的“粘滞性”（hysteresis），避免频繁换目标。

**典型策略：**

* 首先过滤：

  * 必须在前方一定视野区域内（例如屏幕中央 60% 区域）。
* 打分：

  * 越靠近屏幕中心 → 分数越高。
  * `est_distance` 越“近” → 分数越高。
* 维护：

  * 如果当前主目标仍然存在且得分不明显更差 → 保持当前目标。

**核心接口：**

```python
class AttackTargetSelector:
    def __init__(self, center_tolerance_px: int = 200, sticky_frames: int = 15) -> None:
        ...

    def update(self, targets: List[ScreenTarget], screen_center: Tuple[int, int]) -> Optional[ScreenTarget]:
        """根据当前检测到的 targets 更新并返回主攻击目标。"""
        ...
```

---

### 3.3 瞄准控制器（AimController）

**职责：**

* 根据主目标的屏幕坐标，计算“鼠标应朝哪个方向移动”。
* 控制移动速度和缓动，让准心平滑靠近目标中心而非瞬移。

**基本思路：**

* 设屏幕中心为 `(cx, cy)`，目标中心为 `(tx, ty)`。
* 误差：`ex = tx - cx`, `ey = ty - cy`。
* 鼠标移动：`dx = Kp * ex`，`dy = Kp * ey`，并限制单帧最大偏移。
* 当误差在某个“死区”内（例如 5 px）则认为 `is_stable=True`。

**核心接口：**

```python
class AimController:
    def __init__(
        self,
        screen_center: Tuple[int, int],
        kp: float = 0.15,
        max_step_px: int = 25,
        dead_zone_px: int = 5,
    ) -> None:
        ...

    def update(self, target: Optional[ScreenTarget]) -> AimCommand:
        """根据当前主目标，计算本帧需要的鼠标位移。"""
        ...
```

**执行层（不在本控制器内）：**

* `MouseExecutor` 或现有的 `KeyPressManager` 风格组件，真正发送 `dx, dy` 到系统。

---

### 3.4 射击控制器（FireController）

**职责：**

* 根据当前瞄准状态 + 缩圈状态 + 装填状态，决定是否在本帧触发一次“左键点击”。

**输入：**

* `AimCommand.is_stable`（中心对齐情况）。
* 来自“缩圈检测模块”的状态：

  * 初版可以用简单的时间逻辑代替：例如在保持稳定 1.0s 后认为已经瞄准完成。
* 来自“装填状态估计”的状态：

  * 初版可以通过固定装填时间模拟，例如每次开火后冷却 N 秒。

**核心接口：**

```python
class FireController:
    def __init__(
        self,
        min_stable_time: float = 0.8,
        reload_time: float = 8.0,
    ) -> None:
        ...

    def update(
        self,
        dt: float,
        has_target: bool,
        aim_cmd: AimCommand,
    ) -> FireDecision:
        """每帧更新射击状态，决定是否需要开火。"""
        ...
```

**执行层：**

* 调用 `MouseExecutor.click_left()` 或现有控制层，完成一次短按鼠标左键。

---

## 4. AttackModule 总控类

为方便集成到现有 Runtime，提供一个高层封装 `AttackModule`：

```python
class AttackModule:
    """攻击模块高层封装，由 Runtime 在固定 tick 调用。"""

    def __init__(
        self,
        main_view_detector: MainViewDetector,
        target_selector: AttackTargetSelector,
        aim_controller: AimController,
        fire_controller: FireController,
        mouse_executor,
    ) -> None:
        ...

    def update(self, frame_bgr: np.ndarray, dt: float) -> None:
        """在控制线程中被周期性调用。"""
        # 1) 感知：检测主视野目标
        targets = self.main_view_detector.detect(frame_bgr)

        # 2) 目标选择
        best_target = self.target_selector.update(targets, self.aim_controller.screen_center)

        # 3) 瞄准控制
        aim_cmd = self.aim_controller.update(best_target)

        # 4) 执行鼠标移动
        if aim_cmd.dx or aim_cmd.dy:
            self.mouse_executor.move_relative(aim_cmd.dx, aim_cmd.dy)

        # 5) 射击决策
        fire = self.fire_controller.update(
            dt=dt,
            has_target=best_target is not None,
            aim_cmd=aim_cmd,
        )

        # 6) 执行开火
        if fire.should_fire:
            self.mouse_executor.click_left()
```

---

## 5. 与现有系统的集成

### 5.1 与 CaptureService / Runtime 对接

* 在现有导航 Runtime 中，增加一个“攻击线程”或在控制线程中插入攻击逻辑：

  * 从 `CaptureService` 获取主视野截图（和 minimap 截图独立）。
  * 控制 tick 频率可设为 15~25 FPS。

示意：

```python
def _ctrl_loop(self) -> None:
    last_time = time.time()
    while self.running:
        now = time.time()
        dt = now - last_time
        last_time = now

        frame_main = self.capture_service.grab_main_view()
        self.attack_module.update(frame_main, dt)

        time.sleep(interval)
```

### 5.2 与导航模块的关系

* 初版可以简单理解为：

  * 导航模块负责“走到哪里”。
  * 攻击模块负责“看见敌人就打”。
* 二者通过共享的控制层（鼠标/键盘执行器）协同：

  * 移动 / 转向由导航逻辑主导。
  * 鼠标视角可以由攻击模块优先控制（需要未来做仲裁）。

---

## 6. 迭代与扩展方向

1. **加视角状态：**

   * 后续支持右键开镜，增加“狙击视角”专用检测器或共享模型。

2. **缩圈视觉检测：**

   * 当前版本使用“稳定时间”近似缩圈完成；
   * 后续可训练一个小模型专门识别“绿色准星圈大小/状态”。

3. **更复杂的目标优先级：**

   * 优先打残血 / 打近距离 / 打对我开火的敌人。

4. **行为克隆接口：**

   * 攻击模块的输入/输出已经是高度结构化的：

     * 输入：`ScreenTarget` 列表 + 自己状态；
     * 输出：`AimCommand` + `FireDecision`；
   * 后续可以用人类操作日志替换 TargetSelector / AimController / FireController 的部分逻辑。

---

## 7. 实施顺序建议

1. 实现 `MainViewDetector` + demo 脚本，在静态截图上验证目标检测 & 血条关联。
2. 实现 `AttackTargetSelector`：在多目标截图上调试选目标逻辑（可在 debug overlay 中画出当前主目标）。
3. 实现 `AimController`：离线用录屏逐帧测试鼠标 Δx, Δy 输出是否趋近目标中心。
4. 实现 `FireController` 简化版：基于“稳定时间 + 固定装填时间”决策开火。
5. 整合为 `AttackModule`，接入 Runtime，实测一两场对战，开始调整参数。

这样可以始终保持：每一步都能独立调试，每一层都能替换为更智能的版本。
