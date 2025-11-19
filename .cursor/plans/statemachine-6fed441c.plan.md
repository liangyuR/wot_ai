<!-- 6fed441c-8c7d-4b3c-a84e-7abbd7d7a922 1ef1571f-07d6-4fa5-8184-e65626d88406 -->
# 将 StateMachine 改为基于 PaddleOCR OCR 识别

## 目标

将 StateMachine 的状态检测从模板匹配改为使用 PaddleOCR 进行 OCR 文字识别，解决跨分辨率问题，提高准确率。

## 实现方案

### 1. 添加 PaddleOCR 依赖和初始化

- 在 `__init__` 中初始化 PaddleOCR 实例（支持中英文，`lang="ch"`）
- 添加 OCR 相关的成员变量
- 移除模板匹配相关的代码（模板路径、分辨率检测等）

### 2. 定义状态识别关键词

为每个状态定义识别关键词列表：

- **IN_GARAGE**: ["加入战斗", "战斗中"]
- **IN_LOADING**: ["加载", "载入", "LOADING", "加载中"]
- **IN_BATTLE**: 继续使用模板匹配，这部分逻辑稍后我来补充。
- **IN_END**: ["胜利", "失败", "坦克被该玩家击毁", "平局"]

### 3. 实现 OCR 状态检测方法

- 创建 `_ocr_detect_state_()` 方法：
- 接收 frame（numpy 数组）
- 直接使用 numpy 数组
- 调用 PaddleOCR 进行识别
- 遍历所有识别结果，查找关键词匹配
- 返回匹配的状态

### 4. 优化性能

- 考虑是否需要保存临时截图文件，或直接使用 numpy 数组（PaddleOCR 支持）
- 添加 OCR 结果缓存机制（可选，避免重复识别相同画面）
- 保持现有的状态确认机制（连续多帧确认）

### 5. 修改 detect_state 方法

- 移除模板匹配逻辑
- 调用 OCR 检测方法
- 保持相同的返回值和接口

## 文件修改

- `wot_ai/game_modules/core/state_machine.py`：
- 添加 PaddleOCR 导入和初始化
- 定义状态关键词映射
- 实现 OCR 检测方法
- 修改 `detect_state()` 方法
- 移除模板匹配相关代码（可选保留作为备选）

## 注意事项

- 需要安装 paddleocr: `pip install paddleocr`
- OCR 识别可能需要一些时间（10-30ms），但比模板匹配更可靠
- 保持向后兼容，确保接口不变
- 考虑错误处理：OCR 初始化失败时的降级方案

## 待确认问题

1. 每个状态的具体识别关键词是什么？（需要根据实际游戏界面确定）
2. 是否需要保留模板匹配作为备选方案？
3. IN_BATTLE 状态如何识别？（可能需要特殊逻辑）

### To-dos

- [ ] 添加 PaddleOCR 导入和初始化：在 __init__ 中初始化 PaddleOCR 实例，支持中英文识别
- [ ] 定义状态识别关键词：为每个游戏状态（IN_GARAGE, IN_LOADING, IN_BATTLE, IN_END）定义关键词列表
- [ ] 实现 OCR 状态检测方法：创建 _ocr_detect_state_() 方法，使用 PaddleOCR 识别文字并匹配关键词
- [ ] 修改 detect_state 方法：移除模板匹配逻辑，改为调用 OCR 检测方法
- [ ] 清理模板匹配相关代码：移除模板路径、分辨率检测等不再需要的代码（可选保留作为备选）