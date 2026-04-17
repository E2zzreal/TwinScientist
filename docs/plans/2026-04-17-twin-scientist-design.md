# Twin Scientist -- 科研人员数字分身系统架构设计

> 目标：构建一个具有特定科研人员知识、品味、风格的 AI Agent 数字分身，最终能以语音参加学术会议。
>
> 领域：材料科学（氢能催化为主，电池次之，AI/能源交叉）
>
> 日期：2026-04-17

---

## 1. 系统总体架构

采用分层 Agent 架构，共五层：

```
┌─────────────────────────────────────────────────┐
│              交互层 (Interface Layer)             │
│   文字对话 → 语音参会 → 多模态（渐进式开放）        │
├─────────────────────────────────────────────────┤
│            人格层 (Persona Layer)                 │
│   性格模型 · 表达风格 · 科研品味 · 决策偏好         │
├─────────────────────────────────────────────────┤
│            认知层 (Cognition Layer)               │
│   思维框架 · 观点生成 · 类比迁移 · 边界感知         │
├─────────────────────────────────────────────────┤
│            记忆层 (Memory Layer)                  │
│   分层记忆 · 渐进式披露 · 上下文预算管理            │
├─────────────────────────────────────────────────┤
│            数据层 (Data Layer)                    │
│   论文印象 · 认知档案 · 话题索引 · 进化日志         │
└─────────────────────────────────────────────────┘
```

核心思路：不是 RAG 问答，而是通过人格层让 LLM "成为"该科研人员。数据层提供知识基础，记忆层维持连贯性，认知层负责推理，人格层确保风格一致。

---

## 2. 数据层 -- 以"认知印象"替代全文检索

### 2.1 设计原则

放弃向量数据库 + RAG 范式，改为结构化认知档案。人的记忆是经过压缩、带有立场的抽象，不是全文检索。

### 2.2 知识摄入方式

两条路径并行：

**路径 A：自动提取 + 人工确认**

论文/文章经 LLM 提取摘要后，生成针对性问题让用户确认态度和关注点。

**路径 B：交互式初始化**

系统主动提问，逐步构建科研人员画像。例如："你认为氢能最大的瓶颈在哪？"、"哪篇论文对你影响最深？"

### 2.3 认知档案 Schema

```yaml
# 论文印象示例：knowledge/papers/2024-chen-hydrogen-catalyst.yaml
source:
  title: "..."
  authors: ["Chen et al."]
  year: 2024
  field: [氢能, 催化剂]

impression:
  one_sentence: "用MOF衍生碳负载单原子Pt替代传统Pt/C，思路有意思但稳定性数据不够"
  key_takeaway: "单原子分散策略可以大幅降低Pt用量"
  attitude: skeptical_but_interested
  relevance_to_me: high

connections:
  - target: "2023-wang-sac-review"
    relation: "extends"
  - target: "my/research/hoe-catalyst"
    relation: "competing_approach"

memorable_details:
  - "Fig.3的EXAFS数据很漂亮，证实了单原子分散"
  - "但Table 2的循环稳定性只测了100圈，不够"

figure_impressions:
  - figure: "Fig.3 EXAFS fitting"
    saw: "Pt-N配位峰很清晰，没有Pt-Pt峰，确实是单原子"
    judgment: "数据质量高，可信"
  - figure: "Fig.5 polarization curve"
    saw: "过电位比Pt/C低20mV，但电流密度不高"
    judgment: "有改进但幅度有限，实用性存疑"
```

```yaml
# 科研立场示例：persona/research_stance.yaml
core_beliefs:
  - claim: "氢能商业化的瓶颈不在催化剂本身，在系统集成"
    confidence: high
    origin: "多年项目经验"
  - claim: "纯计算筛选催化剂不可靠，必须有实验验证"
    confidence: medium
    origin: "看过太多DFT预测与实验不符的案例"

methodology_preference:
  approach: "实验为主，计算辅助"

taste:
  excited_by: ["新型表征手段", "跨领域迁移的想法", "简洁优雅的实验设计"]
  skeptical_of: ["只有计算没有实验的paper", "过度claim的标题"]
  pet_peeves: ["稳定性测试不充分", "对照实验缺失"]
```

---

## 3. 记忆层 -- 分层记忆与渐进式披露

### 3.1 上下文预算分配

以 128K tokens 窗口为例（预留 28K 给模型推理）：

| 区域 | 预算 | 内容 |
|------|------|------|
| 固定区 | ~8K | System Prompt、L0身份、话题索引、思维框架 |
| 动态区 | ~12K | recall/reflect 工具加载的内容，用完可卸载 |
| 对话区 | ~80K | 对话历史，需压缩管理 |

### 3.2 三层记忆结构

**L0 身份层（永驻上下文，~2K tokens）**

```yaml
# memory/identity.yaml
personality_sketch: |
  说话直接，喜欢用具体实验数据说话。
  对花哨但缺乏实验验证的工作持怀疑态度。
  擅长用类比解释复杂概念。
core_beliefs: [...]
research_focus: [...]
taste_profile: [...]
```

**L1 话题索引（永驻上下文，~2K tokens，指向详情文件）**

```yaml
# memory/topic_index.yaml
topics:
  hydrogen_catalyst:
    summary: "关注Pt替代和单原子催化，偏好实验验证路线"
    paper_count: 12
    stance: "看好但认为稳定性是核心未解决问题"
    detail_files:
      - papers/2024-chen-hydrogen-catalyst.yaml
      - papers/2023-wang-sac-review.yaml
  solid_state_battery:
    summary: "持续跟踪，但非主攻方向，关注界面问题"
    paper_count: 6
    detail_files: [...]
```

**L2 详情档案（按需加载，recall 工具触发）**

各话题的具体论文印象、详细立场、数据点。

### 3.3 渐进式披露检索机制

借鉴 Skills 模式：元数据始终可见，LLM 自己决定何时加载全文。

```
Level 0 -- 始终在场（索引行，~30 tokens/条）
  "hydrogen_catalyst: 关注Pt替代，看好但担忧稳定性"
  → 足够回答泛泛的问题

Level 1 -- recall(topic, depth="summary")，~500 tokens
  加载话题摘要：核心立场 + 关键论点 + 主要分歧
  → 足够回答方向性讨论

Level 2 -- recall(topic, depth="detail"|"specific_paper")，每篇~200 tokens
  加载具体论文印象，只选 2-5 篇相关的
  → 回答具体论文或数据的讨论
```

LLM 本身就是路由器，不需要额外检索工具。topic_index 在上下文中，LLM 天然理解语义匹配。recall 工具本质就是读文件。

### 3.4 对话区压缩策略

```
近期对话（原文保留）    ← 最近 10 轮，完整保留
      │ 当超过阈值时
      ▼
中期摘要（压缩保留）    ← 之前的对话压缩为摘要，压缩比 ~10:1
      │ 当摘要也太长时
      ▼
长期记忆（写入文件）    ← 重要内容持久化到记忆层，对话结束后仍可 recall
```

### 3.5 动态区生命周期

recall 加载的内容在回答完成后标记为"可卸载"。下一轮如果话题切换则释放空间；话题延续则保留。超预算时按 LRU 策略卸载。

### 3.6 会议场景的上下文管理

连续音频流用滑动窗口 + 滚动摘要处理：

```
实时转写 → 只保留最近 5 分钟原文
        → 每 5 分钟将窗口外内容压缩为滚动摘要
        → 维护会议上下文快照（主题/参会者/已讨论话题/当前话题）
        → 始终控制在 ~16K tokens 以内
```

---

## 4. 人格层 -- 多源自动提取 + 交互补盲

### 4.1 三个维度

- **表达风格**：语气、口头禅、解释方式、中英文混用习惯
- **思维模式**：看论文先看什么、评估技术路线的逻辑框架
- **学术品味**：什么让我兴奋、什么让我怀疑、什么让我反感

### 4.2 多源自动提取

| 数据源 | 能提取什么 | 价值 |
|--------|-----------|------|
| 会议录音(转写) | 说话风格、口头禅、即兴反应、反驳/赞同方式、中英文混用 | 最高 |
| 自己的论文 | 学术写作风格、论证结构、常用表述、claim力度 | 高 |
| 个人笔记/文档 | 非正式表达、思考痕迹、关注点 | 中 |
| 阅读过的文章 | 阅读偏好、过滤标准 | 中 |

### 4.3 提取方法

对会议录音转写做风格分析：找最能体现个人风格的发言段落，标注场景类型和风格特征。对多篇论文做对比分析：提取跨论文一致的表述模式和论证偏好。多段录音分别提取后合并去重——只出现一次的特征可能是噪声，多次出现的才是"人格"。

### 4.4 示范对驱动

人格最难用规则描述，但容易通过 few-shot 示范传递：

```yaml
# persona/style.yaml
voice:
  exemplars:
    - context: "被问到一个研究方向的前景"
      bad:  "这个方向有一定的研究价值，值得进一步探索。"
      good: "方向是好的，但你看稳定性数据，100圈就衰减了15%，这离实用差太远。"
      note: "永远锚定在具体数据上，不说空话"

    - context: "评价一篇论文"
      bad:  "这篇文章有一些不足之处，实验设计可以改进。"
      good: "想法挺好的，但control experiment没做干净。你看Fig.3，没有跟bulk对比。"
      note: "先肯定思路，再从实验设计角度精确批评"
```

### 4.5 交互只做两件事

1. **补盲**：数据覆盖不到的维度。"从录音中没找到你评价理论计算工作的例子，你一般怎么评估？"
2. **校准**：自动提取结果的确认/修正。"我总结出你的风格是 [摘要]，像不像？"

---

## 5. 认知层 -- 个人思维框架驱动

### 5.1 核心思想

借鉴丰田 TBP（Toyota Business Practices）：不管遇到什么问题，思考路径是稳定的。科研人员换个陌生课题，评估的"套路"不会变。思维框架本身就是人格的一部分，可迁移到任何新问题。

### 5.2 思维框架（从数据中归纳）

```yaml
# persona/thinking_frameworks.yaml

frameworks:
  evaluate_new_idea:
    name: "评估一个新想法"
    source: "从8段会议讨论中归纳"
    steps:
      - label: "先问动机"
        action: "这个问题为什么重要？解决了什么真实痛点？"
        typical_expression: "你做这个的出发点是什么？"
      - label: "锚定数据"
        action: "有没有实验数据支撑？数据质量如何？"
        typical_expression: "数据在哪？让我看看。"
      - label: "找对标"
        action: "和现有最好的方案比，优势在哪？"
        typical_expression: "你跟XX比过没有？比它好在哪？"
      - label: "追问瓶颈"
        action: "最难的部分是什么？有没有回避关键难题？"
        typical_expression: "这里面最难的一步你打算怎么解决？"
      - label: "判断可行性"
        action: "以现有条件能不能做？成本和周期合不合理？"
        typical_expression: "这个做出来大概要多久？设备够吗？"
      - label: "给结论"
        action: "明确表态，但标注确定程度"
        typical_expression: "我觉得值得试/有点悬，因为..."

  review_a_paper:
    name: "审阅一篇论文"
    steps:
      - label: "看claim"
        action: "标题和摘要声称做到了什么？claim大不大？"
      - label: "翻实验"
        action: "直接跳到实验部分，看数据能否支撑claim"
      - label: "查对照"
        action: "对照实验做了没有？做得干不干净？"
      - label: "看稳定性"
        action: "长期性能数据有没有？测了多少圈？"
      - label: "评新意"
        action: "方法上有没有真正的创新？"
      - label: "关联自己"
        action: "和我的研究有什么关系？能借鉴什么？"

  approach_unfamiliar_topic:
    name: "面对不熟悉的领域"
    steps:
      - label: "类比映射"
        action: "这个领域的问题结构和我熟悉的什么类似？"
        typical_expression: "这个其实和催化里的XX问题是一回事..."
      - label: "找迁移点"
        action: "我的经验哪些可以迁移过去？"
      - label: "识别盲区"
        action: "哪些是我不懂的？"
        typical_expression: "这部分我不确定，你们领域怎么看？"
      - label: "保守表态"
        action: "给观点但明确标注边界"
```

### 5.3 认知运行逻辑

1. 根据问题类型匹配适用的思维框架（可组合多个）
2. 按框架步骤逐步执行，每步中按需 recall 记忆层知识
3. 生成回答，结构隐含框架步骤但表达自然口语化

### 5.4 边界感知

```yaml
# persona/boundaries.yaml
confident_domains:
  - 氢能催化剂（特别是Pt基和单原子催化）
  - 电解水技术
  - 材料表征方法（XRD, EXAFS, TEM）
familiar_but_not_expert:
  - 固态电池
  - 机器学习在材料中的应用
outside_expertise:
  - 有机合成
  - 生物材料
```

confident → 给出明确立场；familiar → 给观点但标注"了解不够深"；outside → "这不是我的方向"。

---

## 6. 交互层 -- 从文字对话到语音参会

### 6.1 演进路线

| 阶段 | 能力 | 时间 |
|------|------|------|
| Phase 1 (MVP) | 文字对话（CLI） | 现在 |
| Phase 2 | 语音对话 + 图像理解 | 3-6月后 |
| Phase 3 | 实时参会 + PPT识别 + 语音克隆 | 6-12月后 |

### 6.2 Phase 1: Agent 主循环

```python
class TwinScientist:
    def __init__(self, config_path):
        self.config = load_yaml(config_path)
        self.context = ContextManager(self.config)
        self.client = anthropic.Client()

    def build_system_prompt(self):
        return "\n".join([
            load_file("persona/identity.yaml"),
            "## 你的知识领域",
            load_file("memory/topic_index.yaml"),
            "## 你的思维方式",
            load_file("persona/thinking_frameworks.yaml"),
            TOOL_INSTRUCTIONS,
        ])

    def chat(self, user_message):
        self.context.prepare(user_message)
        messages = [
            *self.context.get_history(),
            {"role": "user", "content": user_message},
        ]
        while True:
            response = self.client.messages.create(
                model=self.config["model"],
                system=self.build_system_prompt(),
                messages=messages,
                tools=TOOLS,
            )
            if response.stop_reason == "end_turn":
                break
            tool_results = self.execute_tools(response.tool_calls)
            messages.append(response)
            messages.append(tool_results)
        answer = response.content[0].text
        self.context.add_turn(user_message, answer)
        return answer
```

### 6.3 Agent 工具集

```yaml
tools:
  recall:
    description: "回忆特定话题的知识和立场"
    params: { topic, depth: summary|detail|specific_paper }

  reflect:
    description: "对新问题运用思维框架进行推理"
    params: { question, framework_hint? }

  acknowledge_ignorance:
    description: "标记当前话题超出能力边界"
    params: { topic, nearest_known_domain }

  save_to_memory:
    description: "将对话中产生的新观点存入长期记忆"
    params: { topic, content, source: "conversation" }

  see:
    description: "观察并理解一张图片（论文图表、PPT页面）"
    params: { image, context? }
```

### 6.4 Phase 2: 语音对话

在 Phase 1 基础上前后各加一层：语音输入 → STT(Whisper) → Agent → TTS(语音克隆) → 语音输出。Agent 内核不变。

语音克隆推荐 CosyVoice 2（阿里开源）或 GPT-SoVITS，用已有 <10h 录音训练，本地部署保护隐私。

### 6.5 Phase 3: 实时参会

```
会议音频流 + 屏幕共享
      │              │
      ▼              ▼
   实时STT      定时截屏/PPT捕获
      │              │
      ▼              ▼
   会议理解层（融合音频+视觉信息）
      │
      ▼
   触发判断：被点名？涉及我的领域？
      │
      ▼
   Agent生成回答 → 语音克隆TTS → 会议输出
```

额外需要：发言时机判断、会议上下文维护（滑动窗口+滚动摘要）、打断处理。

---

## 7. 持续进化机制

### 7.1 进化来源

1. **新数据摄入**：新论文、新录音
2. **对话中的新观点**：对话过程中产生的新看法
3. **主动反馈**："这个回答不像我"

### 7.2 进化循环

新数据 → 提取 & 对比已有认知 → 分三类处理：知识更新 / 立场演变 / 风格微调 → 记录变更日志。

### 7.3 变更日志

```yaml
# evolution/changelog.yaml
- date: 2025-06-15
  type: stance_update
  topic: hydrogen_catalyst
  before: "看好单原子催化方向"
  after: "对单原子催化的产业化前景更谨慎了"
  source: "2025-06会议录音，提到成本问题"

- date: 2025-07-01
  type: knowledge_expansion
  topic: perovskite_solar
  trigger: "阅读了3篇钙钛矿相关论文"
  initial_stance: "有兴趣但刚开始了解"
```

### 7.4 反馈校正回路

- "这个回答不像我" → 系统追问哪里不像 → 更新 style exemplars
- "我现在不这么看了" → 记录新立场 → 更新 stance + changelog
- "这篇新论文很重要" → 走标准摄入管道生成认知印象

---

## 8. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | Python | ML 生态、快速原型 |
| LLM | Claude API | 长上下文、工具调用原生支持 |
| Agent 框架 | 不用框架，自己写主循环 | 逻辑透明，完全可控 |
| 存储 | YAML 文件 + SQLite | 零运维，可读可编辑 |
| 录音转写 | Whisper (本地) | 隐私、免费、质量足够 |
| 前端(MVP) | 终端 CLI | 最快出活 |
| 语音克隆 | CosyVoice 2 / GPT-SoVITS | 开源、少样本、中文好、本地部署 |
| 图像理解 | Claude Vision | 与主 LLM 统一 |

---

## 9. 项目结构

```
TwinScientist/
├── config.yaml                  # 全局配置
│
├── persona/                     # 人格层
│   ├── identity.yaml            # L0 身份
│   ├── style.yaml               # 表达风格 + 示范对
│   ├── thinking_frameworks.yaml # 思维框架
│   └── boundaries.yaml          # 能力边界
│
├── memory/                      # 记忆层
│   ├── topic_index.yaml         # 话题索引
│   ├── topics/                  # 各话题详情
│   ├── papers/                  # 论文印象
│   └── conversations/           # 对话记忆
│
├── evolution/                   # 进化记录
│   └── changelog.yaml
│
├── ingestion/                   # 数据摄入管道
│   ├── paper_processor.py       # 论文 → 认知印象
│   ├── audio_processor.py       # 录音 → 转写 → 风格提取
│   ├── persona_extractor.py     # 多源归纳人格特征
│   └── interactive_init.py      # 交互式初始化问答
│
├── agent/                       # Agent 核心
│   ├── main.py                  # 主循环
│   ├── context_manager.py       # 上下文窗口管理
│   ├── tools.py                 # recall / reflect / see / save_to_memory
│   └── prompts.py               # System prompt 组装
│
├── multimodal/                  # 多模态能力
│   ├── vision.py                # 图像理解
│   ├── voice_clone/             # 语音克隆
│   │   ├── train.py             # 训练克隆模型
│   │   └── synthesize.py        # TTS 推理
│   └── meeting_vision.py        # 会议屏幕理解
│
└── interface/                   # 交互层
    ├── cli.py                   # Phase 1: 终端对话
    ├── voice.py                 # Phase 2: 语音 I/O
    └── meeting.py               # Phase 3: 会议接入
```

---

## 10. MVP 里程碑

| 里程碑 | 交付物 | 验证标准 | 周期 |
|--------|--------|---------|------|
| M1: 能对话 | CLI 文字对话 | 用该科研人员风格回答；超范围能说"不熟"；20轮不丢上下文 | 1-2周 |
| M2: 能记忆 | recall 三级加载 + 对话压缩 | 话题切换加载/卸载正常；30轮+能引用早期讨论；新观点可持久化 | 1-2周 |
| M3: 能摄入 | 半自动数据摄入管道 | 论文PDF→印象文件；录音→风格提取；交互式校准可修正偏差 | 2-3周 |
| M4: 能进化 | 反馈回路 + 变更日志 | 风格纠正后下次改善；立场更新后一致；变更可审计可回滚 | 1-2周 |
| M5: 能看图 | see 工具 + 图表印象提取 | 理解论文图表并给出专业回应 | 1-2周 |
| M6: 能说话 | 语音克隆 + STT/TTS | 语音对话端到端可用，声音接近本人 | 2-3周 |
| M7: 能参会 | 会议接入 + PPT识别 + 发言时机 | 在真实会议中合理发言 | 3-4周 |
