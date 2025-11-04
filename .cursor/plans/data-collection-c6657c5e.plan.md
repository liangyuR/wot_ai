<!-- c6657c5e-a675-4847-a962-4a08a0f11c54 a86747f3-c0a7-4775-96db-7564c4dcabb0 -->
# YOLO瞄准辅助模块实现

## 目标

在 `yolo/` 目录中创建瞄准辅助系统，实现从YOLO检测到的目标弱点坐标到鼠标移动的完整转换。

## 文件结构

```
yolo/
├── aim_assist.py          # 核心瞄准辅助模块
├── calibration.py          # 标定工具脚本
├── aim_config.yaml        # 瞄准系统配置文件（FOV、灵敏度等）
└── main.py                # 现有YOLO检测脚本
```

## 实现内容

### 1. aim_assist.py - 核心模块

- `AimAssist` 类：
  - 像素偏移 → 角度偏移转换（基于FOV）
  - 角度偏移 → 鼠标移动单位转换（基于标定系数）
  - 完整转换流程：目标坐标 → 鼠标移动
  - 平滑移动（EMA + 最大步长限制）
- `CalibrationTool` 类：
  - 标定系数测量工具
  - 辅助用户进行实际标定

### 2. aim_config.yaml - 配置文件

包含：

- 屏幕分辨率（width, height）
- FOV设置（horizontal_fov, vertical_fov可选）
- 鼠标灵敏度（mouse_sensitivity）
- 标定系数（calibration_factor，可选）
- 平滑参数（smoothing_factor, max_step）

### 3. calibration.py - 标定工具

交互式标定脚本：

- 引导用户进行标定测量
- 计算并保存标定系数到配置文件

### 4. 集成示例

在main.py中添加使用示例，展示如何将YOLO检测结果转换为鼠标移动。

## 技术要点

- 像素到角度转换：使用精确的arctan公式，考虑FOV
- 角度到鼠标单位转换：需要实际标定
- 平滑算法：EMA平滑 + 最大步长

### To-dos

- [x] 架构解耦：拆分 GameplayRecorder，创建独立的屏幕捕获、输入监听、录制循环模块
- [x] 创建屏幕捕获抽象层，实现 DxcamScreenCapture（高性能）和 MssScreenCapture（备选）