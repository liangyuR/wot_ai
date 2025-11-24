# WoT Robot

我真的不想手刷银币

## 项目结构

## 安装

### 使用 Poetry

1. 安装 Poetry（如果尚未安装）：
```bash
pip install poetry
```

2. 安装项目依赖：
```bash
poetry install
```

3. 激活虚拟环境：
```bash
poetry env activate
```

## 使用

## 开发

### 运行测试

### 构建

## 许可证

[添加许可证信息]


20251124
1. 在战斗中被击毁后，导航没有正常关闭（此时的 state 检测貌似被关闭了）
2. 路径规划效果不错，但是运动偏离太大，且超过偏移预警也不去修正。
3. state 检测需要确认3次，没有必要