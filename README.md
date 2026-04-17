# Twin Scientist

科研人员数字分身系统 —— 构建具有特定科研人员知识、品味和风格的 AI Agent。

## 愿景

让数字分身能像真人一样思考和表达：用他的方式评价论文、提出观点、参加学术会议。不是知识检索引擎，而是一个有性格、有记忆、能进化的科研伙伴。

## 架构

```
交互层  →  文字对话 / 语音参会 / 多模态
人格层  →  表达风格 · 科研品味 · 决策偏好
认知层  →  思维框架 · 观点生成 · 边界感知
记忆层  →  分层记忆 · 渐进式披露 · 上下文预算
数据层  →  认知印象 · 话题索引 · 进化日志
```

核心设计决策：
- **认知印象替代 RAG**：存储"对论文的印象和态度"而非原文，像人一样记忆
- **渐进式披露**：借鉴 Skills 模式，LLM 自身作为记忆路由器
- **个人思维框架**：从录音/论文中归纳科研人员的思考套路（类似丰田 TBP）
- **多源人格提取**：从录音、论文、笔记中自动提取风格，交互问答仅用于补盲

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/E2zzreal/TwinScientist.git
cd TwinScientist

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY

# 5. 运行 CLI 对话
python -m interface.cli
```

## 项目结构

```
TwinScientist/
├── persona/                  # 人格层（YAML，可直接编辑）
├── memory/                   # 记忆层（话题索引 + 论文印象）
├── evolution/                # 进化日志
├── ingestion/                # 数据摄入管道
├── agent/                    # Agent 核心（主循环 + 工具 + 上下文管理）
├── multimodal/               # 多模态（图像理解 + 语音克隆）
├── interface/                # 交互层（CLI / 语音 / 会议）
└── tests/                    # 测试
```

## 里程碑

| # | 里程碑 | 状态 |
|---|--------|------|
| M1 | 能对话 — CLI 文字对话，人格风格表达 | 开发中 |
| M2 | 能记忆 — recall 三级加载 + 对话压缩 | 待开始 |
| M3 | 能摄入 — 论文/录音半自动摄入管道 | 待开始 |
| M4 | 能进化 — 反馈回路 + 变更日志 | 待开始 |
| M5 | 能看图 — 论文图表理解 | 待开始 |
| M6 | 能说话 — 语音克隆 + STT/TTS | 待开始 |
| M7 | 能参会 — 实时会议参与 + PPT 识别 | 待开始 |

## 文档

- [架构设计](docs/plans/2026-04-17-twin-scientist-design.md)

## License

MIT
