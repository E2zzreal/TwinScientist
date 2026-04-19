# 数据预处理指南

在将数据摄入 Twin Scientist 之前，需要先对原始数据进行预处理。

---

## 完整流程总览

```
原始数据                    预处理                       摄入
─────────────────────────────────────────────────────────────────
录音文件（mp4/m4a/wav）
  → audio_prep.py          格式转换、分割、质量评估
  → transcript_clean.py    转写清洗、说话人标注       → audio_processor.py

论文 PDF
  → pdf_prep.py            文本提取、元数据识别、      → batch_ingest.py
                            图表提取                   → paper_processor.py

会议字幕（腾讯/飞书）
  → transcript_clean.py    清洗、合并、过滤            → audio_processor.py
```

---

## 脚本详细说明

### 1. audio_prep.py — 录音预处理

**功能：** 格式转换、静音分割、质量评估

**依赖安装：**
```bash
pip install pydub numpy scipy
# 还需要 ffmpeg：
sudo apt-get install ffmpeg        # Linux
brew install ffmpeg                # Mac
```

**典型用法：**
```bash
# 处理单个文件
python preprocess/audio_prep.py \
    --input 组会录音.mp4 \
    --output_dir data/audio/ \
    --speaker 张三

# 批量处理整个目录
python preprocess/audio_prep.py \
    --input recordings/ \
    --output_dir data/audio/ \
    --batch \
    --speaker 张三
```

**输出：**
```
data/audio/
├── 组会录音.wav                  # 转换后的音频
├── 组会录音_chunks/              # 按静音分割的片段
│   ├── 组会录音_chunk001.wav
│   └── ...
├── 组会录音_for_labeling.txt     # 带时间戳的转写（用于说话人标注）
└── preprocessing_report.json    # 质量报告
```

**说话人标注（重要！）**

自动转写后，需要手动标注 `_for_labeling.txt` 文件：
```
# 把 ??? 替换为实际说话人姓名
[00:12] 张三：这是发言内容        # ← 保留
[00:45] 李四：另一段发言          # ← 把 ??? 改为 李四
[01:20] ???：不确定谁说的         # ← 可以删掉这行
```

标注完后，用 `audio_processor.py` 摄入：
```bash
./venv/bin/python -c "
from agent.main import TwinScientist
from ingestion.audio_processor import process_audio
agent = TwinScientist('.')
with open('data/audio/组会录音_for_labeling.txt') as f:
    transcript = f.read()
process_audio(
    client=agent.client,
    model=agent.config['model'],
    persona_dir='persona',
    speaker='张三',
    transcript_text=transcript,
)
"
```

---

### 2. transcript_clean.py — 会议字幕清洗

**功能：** 自动清洗腾讯会议/飞书/Zoom字幕文件

**支持的格式（自动检测）：**
- 腾讯会议导出 `.txt`
- 飞书妙记 `.txt`
- 标准 SRT 字幕 `.srt`
- 通用 `说话人: 内容` 格式

**用法：**
```bash
# 清洗单个文件
python preprocess/transcript_clean.py \
    --input 腾讯会议字幕.txt \
    --speaker 张三 \
    --output data/transcripts/cleaned.txt

# 修正说话人名称（腾讯会议常把名字记成"主持人"等）
python preprocess/transcript_clean.py \
    --input 字幕.txt \
    --speaker 张三 \
    '--speaker_map {"主持人":"张三","与会者1":"李四","与会者2":"王五"}'

# 批量处理
python preprocess/transcript_clean.py \
    --input transcripts/ \
    --batch \
    --output_dir data/transcripts_cleaned/ \
    --speaker 张三
```

**实际清洗效果示例：**
```
清洗前（腾讯会议自动字幕）：
  主持人
  00:01:23 那个那个那个这个稳定性它其实啊你看这个
  00:01:25 数据100圈就衰减了

清洗后：
  [00:01] 张三：这个稳定性你看这个数据100圈就衰减了
```

---

### 3. pdf_prep.py — PDF 批量预处理

**功能：** 提取文本、识别元数据、可选图表提取

**依赖安装：**
```bash
pip install pypdf pdfplumber

# 图表提取（可选）：
pip install pdf2image pillow
sudo apt-get install poppler-utils    # Linux
brew install poppler                   # Mac
```

**用法：**
```bash
# 基础：批量处理论文目录
python preprocess/pdf_prep.py \
    --input papers/ \
    --output_dir data/papers_processed/ \
    --field 氢能 催化剂 单原子催化

# 使用 Zotero/Mendeley 导出的 BibTeX 补充元数据（强烈推荐）
python preprocess/pdf_prep.py \
    --input papers/ \
    --bibtex refs.bib \
    --output_dir data/papers_processed/ \
    --speaker 张三

# 同时提取图表图像（用于视觉分析）
python preprocess/pdf_prep.py \
    --input papers/ \
    --extract_figures \
    --output_dir data/papers_processed/
```

**从 Zotero 导出 BibTeX：**
1. 选择你的文献库 → 右键 → "导出文件库"
2. 格式选 "BibTeX"
3. 导出为 `refs.bib`

**输出：**
```
data/papers_processed/
├── texts/                        # 提取的文本
│   ├── chen2024_sac.txt
│   └── ...
├── figures/                      # 图表图像（如果开启）
│   └── chen2024_sac/
│       ├── page01.png
│       └── ...
└── ingestion_manifest.json       # 摄入清单（供 batch_ingest.py 使用）
```

---

### 4. batch_ingest.py — 批量摄入

**功能：** 读取 `ingestion_manifest.json`，自动为每篇论文生成认知印象

```bash
# 基础摄入
python preprocess/batch_ingest.py \
    --manifest data/papers_processed/ingestion_manifest.json

# 指定话题（自动更新 topic_index.yaml）
python preprocess/batch_ingest.py \
    --manifest data/papers_processed/ingestion_manifest.json \
    --topic hydrogen_catalyst

# 重新摄入已处理的文件
python preprocess/batch_ingest.py \
    --manifest data/papers_processed/ingestion_manifest.json \
    --no_skip
```

---

## 推荐工作流（完整示例）

### 场景：拿到10篇论文PDF + 3段组会录音

```bash
# Step 1: 处理 PDF
python preprocess/pdf_prep.py \
    --input ~/Downloads/papers/ \
    --bibtex ~/Zotero/refs.bib \
    --output_dir data/papers/ \
    --field 氢能 催化剂 \
    --speaker 张三

# Step 2: 批量摄入论文
python preprocess/batch_ingest.py \
    --manifest data/papers/ingestion_manifest.json \
    --topic hydrogen_catalyst

# Step 3: 处理录音
python preprocess/audio_prep.py \
    --input ~/recordings/ \
    --output_dir data/audio/ \
    --speaker 张三

# Step 4: 手动标注说话人（编辑 *_for_labeling.txt 文件）
# 把 ??? 改为实际说话人姓名

# Step 5: 摄入录音（提取风格）
./venv/bin/python -c "
import os, glob
from agent.main import TwinScientist
from ingestion.audio_processor import process_audio

agent = TwinScientist('.')
for txt_file in glob.glob('data/audio/*_for_labeling.txt'):
    with open(txt_file) as f:
        transcript = f.read()
    process_audio(
        client=agent.client,
        model=agent.config['model'],
        persona_dir='persona',
        speaker='张三',
        transcript_text=transcript,
    )
    print(f'处理完成: {os.path.basename(txt_file)}')
"

# Step 6: 交互式补盲（填写录音中没有覆盖的维度）
./venv/bin/python -c "
from ingestion.interactive_init import run_interactive_init
run_interactive_init('persona')
"

# Step 7: 开始对话，验证效果
./venv/bin/python -m interface.cli
```

---

## 数据质量检查清单

在摄入前，确认以下几点：

**录音：**
- [ ] 信噪比 > 20dB（quality_report 中显示"好"或"中"）
- [ ] 目标发言人的发言占比 > 30%
- [ ] 已完成说话人标注（没有 ??? 残留）
- [ ] 包含评价性发言（不只是描述性陈述）

**论文：**
- [ ] 文本提取 > 500 词（不是扫描版PDF）
- [ ] 元数据识别正确（标题、作者、年份）
- [ ] 这些是该科研人员真正深度阅读过的论文（不是领域综述式泛读）

**整体：**
- [ ] 至少3种不同场景的录音（评论论文/回应问题/解释给外行听）
- [ ] 论文覆盖主要研究方向，而非全都集中在一个子话题
