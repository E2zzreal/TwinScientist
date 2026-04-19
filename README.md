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
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API（编辑 config.yaml 填入 base_url 和 model）
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY=your-key

# 5. 运行文字对话
python -m interface.cli

# 6. 运行语音对话（需额外安装 openai-whisper）
python -c "from interface.voice import run_voice_cli; run_voice_cli('.')"

# 7. 会议参与（转写文件模式）
python -c "
from agent.main import TwinScientist
from interface.meeting import run_meeting_from_audio
agent = TwinScientist('.')
run_meeting_from_audio(agent, 'transcript.txt', twin_name='张三',
    confident_domains=['氢能催化剂', '电解水'])
"
```

## 项目结构

```
TwinScientist/
├── config.yaml              # 配置（API provider / model / base_url）
├── persona/                 # 人格层（YAML，可直接编辑）
├── memory/                  # 记忆层（话题索引 + 论文印象 + 对话记忆）
├── evolution/               # 进化日志（自动记录风格和立场变更）
├── ingestion/               # 数据摄入（PDF / 录音 / 交互式校准）
├── agent/                   # Agent 核心（主循环 / 工具 / 上下文管理 / 进化）
├── multimodal/              # 多模态（图像理解 / STT / TTS / 屏幕截图）
├── interface/               # 交互层（CLI / 语音 / 会议）
├── docs/                    # 文档（架构设计 / 使用指南）
└── tests/                   # 自动化测试（100 个测试）
```

## 里程碑

| # | 里程碑 | 核心能力 | 状态 |
|---|--------|---------|------|
| M1 | 能对话 | CLI 文字对话，人格风格表达 | ✅ 完成 |
| M2 | 能记忆 | LLM智能压缩 + 跨会话持久化 | ✅ 完成 |
| M3 | 能摄入 | PDF/录音/交互式摄入管道 | ✅ 完成 |
| M4 | 能进化 | 风格校正 + 立场更新 + 变更日志 | ✅ 完成 |
| M5 | 能看图 | 论文图表 / 实验截图科学分析 | ✅ 完成 |
| M6 | 能说话 | STT(Whisper) + TTS(edge-tts) 语音对话 | ✅ 完成 |
| M7 | 能参会 | 实时会议 + PPT识别 + 发言时机判断 | ✅ 完成 |

## 支持的 API 服务商

| 服务商 | provider | 示例 model |
|--------|----------|-----------|
| DeepSeek | `openai_compatible` | `deepseek-chat` |
| 阿里云百炼（Qwen） | `openai_compatible` | `qwen-plus` |
| 智谱 AI | `openai_compatible` | `glm-4` |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` |
| 本地 Ollama | `openai_compatible` | `qwen2.5:7b` |

## 文档

- [使用指南（非技术背景）](docs/USER_GUIDE.md)
- [架构设计文档](docs/plans/2026-04-17-twin-scientist-design.md)

## License

MIT
