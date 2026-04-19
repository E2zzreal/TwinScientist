# Twin Scientist 使用指南

> 面向科研人员的数字分身系统 —— 无需编程背景即可上手

---

## 这是什么？

Twin Scientist 可以把你的科研知识、思维方式和说话风格"复刻"成一个 AI 对话助手。

**它能做什么：**
- 用你自己的方式评价论文、回答学术问题
- 记住你看过的文章、形成的观点
- 随时间不断进化，越用越像你

**它不能做什么：**
- 它不会替你做实验
- 它不会自动搜索最新文献
- 它的知识来源于你主动提供的材料

---

## 目录

1. [第一次安装](#第一次安装)
2. [配置 API 服务](#配置-api-服务)
3. [填写你的基本档案](#填写你的基本档案)
4. [添加论文知识](#添加论文知识)
5. [添加会议录音](#添加会议录音)
6. [开始对话](#开始对话)
7. [分析论文图表](#分析论文图表)
8. [语音对话](#语音对话)
9. [参加会议](#参加会议)
10. [纠正和进化](#纠正和进化)
11. [常见问题](#常见问题)

---

## 第一次安装

### 你需要准备

- 一台装有 Python 的电脑（Windows / Mac / Linux 均可）
- 一个 AI API 账号（如 DeepSeek、Qwen 等，需要自行申请）
- 项目文件（从 GitHub 下载）

### 步骤

**1. 下载项目**

打开终端（Windows 用命令提示符或 PowerShell，Mac/Linux 用 Terminal），输入：

```bash
git clone https://github.com/E2zzreal/TwinScientist.git
cd TwinScientist
```

**2. 创建运行环境**

```bash
python3 -m venv venv
```

激活环境：
- Mac / Linux：`source venv/bin/activate`
- Windows：`venv\Scripts\activate`

激活后，命令行最前面会出现 `(venv)` 字样。

**3. 安装依赖**

```bash
pip install -r requirements.txt
```

这一步需要几分钟，有网络才能完成。

---

## 配置 API 服务

系统通过 AI API 来驱动对话。你需要有一个 API 账号，并把密钥告诉系统。

### 第一步：填写 API 配置

打开项目根目录的 `config.yaml` 文件（可以用记事本、VS Code 或任意文本编辑器打开）。

找到以下几行，把占位符替换成你自己的值：

```yaml
provider: openai_compatible
base_url: YOUR_BASE_URL    ← 改成你的 API 地址
model: YOUR_MODEL_ID       ← 改成你要用的模型名
```

**常见平台填写示例：**

| 平台 | base_url | model |
|------|----------|-------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里云百炼（Qwen） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |

### 第二步：设置 API Key

**不要把 Key 直接写进 config.yaml**（避免泄露）。

把 Key 存到环境变量中。每次打开终端后运行一次：

```bash
export OPENAI_API_KEY=你的API密钥
```

或者把这行加入你的 `~/.bashrc` / `~/.zshrc`，就不用每次手动输入了。

---

## 填写你的基本档案

系统需要了解"你是谁"才能模仿你的风格。打开 `persona/identity.yaml`，用文本编辑器填写以下内容：

```yaml
personality_sketch: |
  在这里用几句话描述你的说话风格。
  例如：说话直接，喜欢用具体数据说话。
  不喜欢模糊表达，有观点会直接说出来。

core_beliefs:
  - claim: "你最确信的一个学术观点"
    confidence: high
  - claim: "另一个观点，但没那么确定"
    confidence: medium

research_focus:
  - "你的主要研究方向，如：氢能催化剂"
  - "次要方向，如：固态电池"

taste_profile:
  excited_by:
    - "让你感到兴奋的研究类型，如：新型表征手段"
    - "另一种，如：跨领域迁移的想法"
  skeptical_of:
    - "让你持怀疑态度的东西，如：只有计算没有实验的paper"
```

填完保存即可，不需要重启系统，下次对话自动生效。

---

## 添加论文知识

系统不直接"阅读"整篇论文，而是存储你对论文的**印象和评价**——就像你自己记笔记一样。

### 方式一：自动处理 PDF（推荐）

如果你有论文的 PDF 文件，可以用以下命令自动提取：

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from ingestion.paper_processor import process_paper

agent = TwinScientist('.')
process_paper(
    client=agent.client,
    model=agent.config['model'],
    pdf_path='你的论文.pdf',          # 换成实际路径
    memory_dir='memory',
    metadata={
        'title': '论文标题',
        'authors': ['作者姓名'],
        'year': 2024,
        'field': ['氢能', '催化剂'],
    }
)
print('完成')
"
```

运行后，系统会用 AI 阅读论文，生成印象文件，存到 `memory/papers/` 目录。

### 方式二：手动填写（更准确）

在 `memory/papers/` 目录下新建一个 YAML 文件，文件名格式为 `年份-关键词.yaml`，例如 `2024-chen-sac.yaml`：

```yaml
source:
  title: "论文完整标题"
  authors: ["第一作者", "通讯作者"]
  year: 2024
  field: [氢能, 催化剂, 单原子催化]

impression:
  one_sentence: "一句话总结这篇论文，加上你的评价"
  key_takeaway: "这篇论文最值得记住的一个点"
  attitude: skeptical_but_interested   # 从以下选一个：
                                       # excited / interested / neutral
                                       # skeptical_but_interested / skeptical / critical
  relevance_to_me: high               # high / medium / low

memorable_details:
  - "你记住的具体数据点，如：Fig.3的EXAFS数据，Pt-N峰清晰"
  - "另一个细节，如：但稳定性只测了100圈，太少"

connections:
  - target: "相关论文的文件名（不含.yaml）"
    relation: "关系类型，如 extends / competing_approach / inspired_by"
```

### 更新话题索引

添加完论文后，需要在 `memory/topic_index.yaml` 中登记这个话题（如果还没有的话）：

```yaml
topics:
  hydrogen_catalyst:              # 话题ID，英文，下划线分隔
    summary: "一句话描述你在这个方向的关注点"
    paper_count: 5                # 这个话题下有几篇论文（大概数字即可）
    stance: "你对这个方向的总体立场"
    detail_files:
      - papers/2024-chen-sac.yaml  # 列出相关论文的文件名
      - papers/2023-wang-review.yaml
```

---

## 添加会议录音

录音是人格提取质量最高的来源，推荐优先处理。

### 准备录音文件

支持格式：`.mp3`、`.wav`、`.m4a` 等常见音频格式。

建议选择发言人说话清晰、内容与学术讨论相关的片段，时长 10 分钟到数小时均可。

### 安装语音识别（首次使用）

```bash
pip install openai-whisper
```

注意：这个安装包较大（约 1-2GB），需要一些时间。

### 处理录音

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from ingestion.audio_processor import process_audio

agent = TwinScientist('.')
result = process_audio(
    client=agent.client,
    model=agent.config['model'],
    persona_dir='persona',
    speaker='张三',                    # 换成发言人的名字
    audio_path='会议录音.mp3',          # 换成实际文件路径
    save_transcript_to='transcripts/会议录音.txt',   # 可选，保存转写文本
)
print(f'提取了 {result[\"exemplars_added\"]} 个风格示范')
"
```

处理完成后，系统会自动把提取出的风格特征更新到 `persona/style.yaml`。

### 如果只有转写文本

如果你已经有会议的文字转写（比如腾讯会议、飞书的自动字幕），可以跳过语音识别，直接用文本：

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from ingestion.audio_processor import process_audio

agent = TwinScientist('.')
with open('会议转写.txt', 'r') as f:
    transcript = f.read()

process_audio(
    client=agent.client,
    model=agent.config['model'],
    persona_dir='persona',
    speaker='张三',                    # 换成发言人的名字
    transcript_text=transcript,
)
print('完成')
"
```

---

## 开始对话

一切准备好后，在终端输入：

```bash
./venv/bin/python -m interface.cli
```

界面会显示一个对话框，你可以直接输入问题。

### 可用的特殊命令

| 命令 | 作用 |
|------|------|
| `/quit` | 退出对话，自动保存本次会话摘要 |
| `/status` | 查看当前记忆占用情况 |

### 对话技巧

**普通问答：** 直接提问，系统会用数字分身的风格回答。

```
你：你怎么看最近单原子催化在HER上的进展？
分身：好，说说我的看法……
```

**讨论具体论文：** 提到论文相关信息，系统会自动调用记忆。

```
你：Chen 2024那篇MOF衍生碳的工作你怎么评价？
分身：（自动加载该论文的印象，给出有依据的评价）
```

**超出范围的问题：** 数字分身会诚实说"这不是我的方向"。

---

## 分析论文图表

数字分身可以直接看懂 XRD、SEM、TEM、极化曲线等科研图像，并以你的眼光给出评价。

> **前提：** 你使用的 API 服务商需要支持视觉模型，例如 `qwen-vl-plus`、`gpt-4o`、`claude-sonnet-4-20250514`。在 `config.yaml` 的 `model` 字段填写支持视觉的型号。

### 在对话中发送图片

启动 CLI 对话后，直接告诉分身图片的路径：

```
你：帮我看一下这张图  /path/to/xrd_pattern.png  这是氢能催化剂的XRD图谱
分身：（用科研人员的眼光分析图像，关注晶相、峰位、杂峰等）
```

### 批量处理论文图表

在处理论文 PDF 时，也可以同时提取图表印象：

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from ingestion.paper_processor import extract_figure_impressions

agent = TwinScientist('.')
impressions = extract_figure_impressions(
    client=agent.client,
    image_paths=['fig1.png', 'fig2.png', 'fig3.png'],
    paper_title='你的论文标题',
)
for imp in impressions:
    print(imp)
"
```

---

## 语音对话

不想打字？可以直接说话。

### 安装语音依赖（首次使用）

```bash
pip install openai-whisper sounddevice soundfile numpy
```

注意：`openai-whisper` 安装包较大（约 1-2GB），需要一些时间。

### 语音文件转换

如果你有录制好的音频文件，可以处理成语音回答：

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from interface.voice import process_voice_file

agent = TwinScientist('.')
process_voice_file(
    agent=agent,
    audio_input_path='my_question.wav',   # 你的提问音频
    audio_output_path='response.mp3',     # 分身的语音回答
    voice='zh-CN-YunxiNeural',            # 合成声音（见下方列表）
)
"
```

### 实时麦克风对话

```bash
./venv/bin/python -c "
from interface.voice import run_voice_cli
run_voice_cli('.', voice='zh-CN-YunxiNeural', duration=5)
"
```

程序启动后，按 Enter 开始录音（默认 5 秒），说完后自动转写并生成语音回答。输入 `q` 退出。

### 可用的中文语音

| 声音名称 | 性别 | 风格 |
|---------|------|------|
| `zh-CN-YunxiNeural` | 男 | 沉稳（推荐）|
| `zh-CN-YunjianNeural` | 男 | 活跃 |
| `zh-CN-XiaoxiaoNeural` | 女 | 温和 |
| `zh-CN-XiaoyiNeural` | 女 | 活泼 |

在 `run_voice_cli` 的 `voice` 参数中填写任一名称即可切换。

---

## 参加会议

数字分身可以"旁听"会议，在被点名或讨论到你的领域时自动发言。

### 方式一：处理会议转写文件（推荐入门）

如果你有腾讯会议、飞书等平台的自动字幕文件，先整理成如下格式保存为 `.txt`：

```
张三：今天我们讨论一下氢能催化剂的最新进展。
李四：最近单原子催化的文章很多，请问XXX老师你怎么看？
王五：我认为稳定性问题还没有解决……
```

然后运行：

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from interface.meeting import run_meeting_from_audio

agent = TwinScientist('.')
responses = run_meeting_from_audio(
    agent=agent,
    transcript_path='meeting_transcript.txt',
    twin_name='XXX',                              # 换成发言人的名字
    confident_domains=['氢能催化剂', '电解水', '单原子催化'],
)
print(f'共发言 {len(responses)} 次')
for r in responses:
    print(f'触发：{r[\"trigger\"]}')
    print(f'回答：{r[\"response\"][:100]}')
    print()
"
```

系统会在以下情况自动发言：
- 被点名（如"XXX 老师你怎么看"）
- 话题涉及你的专业领域

### 方式二：实时麦克风参会

```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from interface.meeting import run_realtime_meeting

agent = TwinScientist('.')
run_realtime_meeting(
    agent=agent,
    twin_name='XXX',
    confident_domains=['氢能催化剂', '电解水'],
    record_duration=3,   # 每次录 3 秒后判断是否需要发言
)
"
```

程序会持续监听麦克风，自动识别发言内容，在合适时机生成并播放语音回答。按 Ctrl+C 退出。

### 会议中的 PPT 识别

在会议期间，可以同时截取屏幕内容，让分身了解当前展示的材料：

```bash
# 这段代码在会议主循环中自动触发，也可手动调用
./venv/bin/python -c "
from agent.main import TwinScientist
from multimodal.meeting_vision import analyze_screen_content
from PIL import ImageGrab

agent = TwinScientist('.')
# 截取当前屏幕
screenshot = ImageGrab.grab()
screenshot.save('/tmp/screen.png')

result = analyze_screen_content(
    client=agent.client,
    image_path='/tmp/screen.png',
    meeting_topic='氢能催化剂进展讨论',
)
print(result)
"
```

---

## 纠正和进化

数字分身不可能一开始就完全准确。用以下方式告诉它哪里不像你：

### 纠正说话风格

如果你觉得某个回答"不像你说话"，直接告诉它：

```
你：刚才那个回答不像我，我不会说"有一定研究价值"，我会直接说数据
```

系统会把这个反馈记录下来，生成一个新的风格示范，下次对话就会改进。

### 更新学术立场

如果你的观点随时间变化了：

```
你：我现在对单原子催化的看法变了，产业化时间线比我之前想的要长很多
```

系统会更新你在这个话题上的立场，以后回答相关问题时会反映新观点。

### 查看进化记录

所有的修改都记录在 `evolution/changelog.yaml` 中，可以随时打开查看历史变更。如果某次修改效果不好，可以手动删除对应的条目，再把 `persona/style.yaml` 恢复到之前的版本。

---

## 常见问题

**Q：运行时报 "OPENAI_API_KEY not set" 错误**

需要先设置环境变量：
```bash
export OPENAI_API_KEY=你的密钥
```
然后再运行程序。每次打开新的终端窗口都需要重新设置，除非你把它加入了系统配置文件。

---

**Q：对话回复很慢**

这取决于你使用的 AI 服务的响应速度，和系统本身无关。可以尝试换用更快的模型（在 `config.yaml` 的 `model` 字段修改）。

---

**Q：分身说话不像目标科研人员**

这通常是因为 `persona/identity.yaml` 填写不够具体，或者还没有处理录音。建议：
1. 先运行交互式初始化，回答一系列问题：
   ```bash
   ./venv/bin/python -c "
   from ingestion.interactive_init import run_interactive_init
   run_interactive_init('persona')
   "
   ```
2. 提供至少一段 30 分钟以上的会议录音
3. 通过对话中的反馈逐步纠正

---

**Q：记忆里加了论文，但对话中没有体现**

检查两件事：
1. `memory/topic_index.yaml` 中是否有对应话题，且 `detail_files` 列表里包含了该论文的文件名
2. 论文文件是否在 `memory/papers/` 目录下，且文件名与 topic_index 中一致

---

**Q：如何备份我的数字分身数据**

把以下目录复制到安全位置即可：
- `persona/`（人格档案）
- `memory/`（知识记忆）
- `evolution/`（进化日志）

这三个目录存储了所有关键信息，代码本身可以从 GitHub 重新下载。

---

**Q：可以给别人用吗？**

可以，把整个项目文件夹发给对方，对方按本指南安装配置即可。但 `persona/` 和 `memory/` 目录包含个人学术信息，分享前请确认当事人同意。

---

**Q：图像分析提示"视觉功能不支持"**

当前配置的模型不支持图像输入。需要在 `config.yaml` 中换用支持视觉的模型：

```yaml
model: qwen-vl-plus        # 阿里云 Qwen 视觉版
# 或
model: gpt-4o              # OpenAI
# 或
model: claude-sonnet-4-20250514  # Anthropic（需配合 provider: anthropic）
```

---

**Q：语音对话没有声音输出**

检查以下几点：
1. 是否安装了 `sounddevice`：`pip install sounddevice soundfile`
2. 系统音量是否开启
3. Linux 用户可能需要：`sudo apt-get install portaudio19-dev`

---

**Q：实时参会时响应太慢跟不上会议节奏**

这主要取决于 API 响应速度。建议：
1. 换用响应更快的模型（如 DeepSeek 的轻量模型）
2. 增大 `record_duration` 减少处理频次
3. 实时模式更适合专题讨论，不适合快节奏的头脑风暴

---

## 文件结构速查

```
TwinScientist/
├── config.yaml              ← 修改这里配置 API（provider/base_url/model）
├── .env                     ← 存放 API Key（自己创建）
├── persona/
│   ├── identity.yaml        ← 修改这里填写基本信息
│   ├── style.yaml           ← 自动更新，也可手动添加示范
│   ├── thinking_frameworks.yaml  ← 思维框架，自动更新
│   └── boundaries.yaml     ← 专业领域边界
├── memory/
│   ├── topic_index.yaml     ← 修改这里添加话题
│   ├── papers/              ← 论文印象文件放这里
│   ├── topics/              ← 话题详情文件放这里
│   └── conversations/       ← 对话记忆，自动生成
├── evolution/
│   └── changelog.yaml       ← 进化日志，自动记录
├── interface/
│   ├── cli.py               ← 文字对话（python -m interface.cli）
│   ├── voice.py             ← 语音对话
│   └── meeting.py           ← 会议参与
└── multimodal/
    ├── vision.py            ← 图像理解
    ├── stt.py               ← 语音转文字（Whisper）
    ├── tts.py               ← 文字转语音（edge-tts）
    └── meeting_vision.py    ← 会议屏幕分析
```
