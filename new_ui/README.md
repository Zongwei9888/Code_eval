# Code Eval v4.0 - NiceGUI Modern UI

## 🚀 概述

Code Eval v4.0 是一个基于 NiceGUI 构建的现代化 AI 代码分析系统界面。它提供了美观的赛博朋克风格主题、实时流式更新、多智能体工作流可视化等先进功能。

## ✨ 新功能特性

### 1. 现代化 UI 设计
- **赛博朋克主题**: 深色背景配合霓虹色调的高对比度设计
- **玻璃态卡片**: 带模糊效果的半透明卡片组件
- **流畅动画**: 悬浮效果、脉冲动画、渐入效果
- **自定义字体**: Orbitron 标题 + JetBrains Mono 代码字体

### 2. 单文件分析模式
- 项目/文件选择器
- CodeMirror 代码编辑器（语法高亮）
- 实时日志流显示
- 自动修复循环
- 代码差异对比视图

### 3. 快速扫描
- 无需 LLM 的本地 AST 分析
- 语法错误检测
- 项目结构可视化
- 文件树组件

### 4. 多智能体工作流
- **可视化工作流**: 实时显示 Scanner → Analyzer → Fixer → Executor → Reporter 流程
- **步骤追踪**: 每个智能体的状态和进度
- **聊天式日志**: 每个智能体的输出以对话形式展示
- **反馈循环**: 自动重试直到成功

### 5. AI 聊天助手 (新功能)
- 交互式代码问答
- 快速操作按钮（查找 bug、优化、测试等）
- 上下文感知对话
- 代码预览面板

### 6. 高级功能面板
- **代码补全建议**: AI 驱动的代码建议
- **终端模拟器**: 内嵌命令行界面
- **代码生成器**: 自然语言到代码转换
- **Linting 面板**: 实时代码质量检查
- **Git 集成**: 版本控制操作

## 📁 文件结构

```
new_ui/
├── __init__.py          # 模块入口
├── app.py               # 主应用程序
├── theme.py             # 主题配置（颜色、CSS）
├── components.py        # 可重用 UI 组件
├── advanced_features.py # 高级功能组件
├── pages/
│   ├── __init__.py
│   ├── dashboard.py     # 仪表板页面
│   └── workspace.py     # IDE 工作区页面
└── README.md            # 本文档
```

## 🎨 主题配色

| 颜色 | 用途 |
|------|------|
| `#00d4ff` | 主色调（青色霓虹） |
| `#7c3aed` | 次要色（紫色） |
| `#00ff88` | 成功状态（矩阵绿） |
| `#ffaa00` | 警告状态（琥珀色） |
| `#ff3366` | 错误状态（热粉色） |

### 智能体颜色
- Scanner: 青色 `#00d4ff`
- Analyzer: 琥珀色 `#ffaa00`
- Fixer: 绿色 `#00ff88`
- Executor: 蓝色 `#3b82f6`
- Reporter: 品红色 `#d946ef`

## 🛠️ 组件列表

### 基础组件
- `StatusIndicator`: 动画状态指示器
- `MetricCard`: 数据指标卡片
- `ChatMessage`: 智能体聊天气泡
- `FileTree`: 交互式文件树
- `LogViewer`: 实时日志查看器
- `WorkflowVisualizer`: 工作流进度可视化
- `CodeDiffViewer`: 代码差异对比
- `AIPromptInput`: AI 提示输入框

### 高级组件
- `CodeCompletionPanel`: 代码补全建议
- `TerminalEmulator`: 终端模拟器
- `ProjectHealthDashboard`: 项目健康仪表板
- `ConversationMemory`: 对话记忆管理
- `CodeGenerationPanel`: 代码生成面板
- `GitIntegrationPanel`: Git 集成面板
- `LintingPanel`: 代码质量面板
- `KeyboardShortcutsPanel`: 快捷键参考

## 🚀 启动方式

### 方式一：使用启动脚本
```bash
python run_new_ui.py
```

### 方式二：带参数启动
```bash
python run_new_ui.py --host 0.0.0.0 --port 8080 --reload
```

### 方式三：作为模块导入
```python
from new_ui import run_app
run_app(host="127.0.0.1", port=8080)
```

## 📦 依赖安装

```bash
pip install nicegui>=2.0.0
```

或使用完整依赖：
```bash
pip install -r requirements.txt
```

## 🎯 功能路线图

### 已完成 ✅
- [x] 基础 UI 框架
- [x] 主题系统
- [x] 单文件分析
- [x] 快速扫描
- [x] 多智能体工作流
- [x] 日志可视化
- [x] 代码编辑器

### 进行中 🔄
- [ ] 完整 AI 对话集成
- [ ] 代码生成功能
- [ ] Git 集成

### 计划中 📋
- [ ] 实时协作
- [ ] 项目模板
- [ ] 插件系统
- [ ] 自定义主题
- [ ] 移动端适配

## 💡 设计理念

### 1. Coding Agent 场景优化
- 展示智能体思考过程
- 可视化工具调用
- 实时反馈循环

### 2. 开发者体验优先
- 键盘快捷键支持
- 快速操作按钮
- 上下文感知提示

### 3. 可扩展架构
- 组件化设计
- 主题可定制
- 易于集成新功能

## 🔧 自定义配置

### 修改主题颜色
编辑 `theme.py` 中的 `COLORS` 字典：

```python
COLORS = {
    "primary": "#your-color",
    "success": "#your-color",
    # ...
}
```

### 添加新组件
在 `components.py` 中创建新类：

```python
class MyComponent:
    def __init__(self):
        self._build()
    
    def _build(self):
        with ui.element('div').classes('glass-card'):
            # 你的组件内容
            pass
```

## 📝 许可证

MIT License

---

**Code Eval v4.0** - 让 AI 编程助手更加智能和美观 🚀

