#!/usr/bin/env python3
"""
Smoke test: verifies end-to-end agent flow with real API calls.
Run: ANTHROPIC_API_KEY=sk-ant-xxx ./venv/bin/python smoke_test.py
"""
import os
import sys

# Verify API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set.")
    print("Run: export ANTHROPIC_API_KEY=sk-ant-xxxxx")
    sys.exit(1)

from agent.main import TwinScientist

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
agent = TwinScientist(PROJECT_DIR)

TESTS = [
    # (description, question, keywords_expected_in_response)
    (
        "基础身份问答",
        "你好，简单介绍一下你的研究方向",
        ["氢能", "催化"],
    ),
    (
        "领域立场测试",
        "你怎么看单原子催化剂在HER上的进展？",
        ["稳定性", "单原子"],
    ),
    (
        "recall 工具触发",
        "你对Chen 2024那篇MOF衍生碳单原子Pt的论文有什么看法？",
        ["100圈", "稳定性"],
    ),
    (
        "超出边界测试",
        "你对有机合成有什么了解？",
        ["不是", "不太"],  # should admit it's outside expertise
    ),
    (
        "风格测试（应用数据说话）",
        "纯DFT计算筛选催化剂靠不靠谱？",
        ["实验", "验证"],
    ),
]

print("=" * 60)
print("Twin Scientist Smoke Test")
print("=" * 60)

passed = 0
failed = 0

for desc, question, keywords in TESTS:
    print(f"\n[TEST] {desc}")
    print(f"  Q: {question}")
    try:
        answer = agent.chat(question)
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        hit = any(kw in answer for kw in keywords)
        if hit:
            print(f"  ✓ PASS (包含关键词之一: {keywords})")
            passed += 1
        else:
            print(f"  ✗ SOFT FAIL (未找到关键词: {keywords}，但回答本身请人工判断)")
            failed += 1
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        failed += 1

print("\n" + "=" * 60)
print(f"结果: {passed} passed, {failed} soft-failed")
print("注意: soft-fail 不代表错误，请人工阅读回答判断风格是否准确")
print("=" * 60)

# Test give_feedback tool
print("\n[TEST] give_feedback — 立场更新")
try:
    result = agent._execute_tool("give_feedback", {
        "feedback_type": "stance",
        "topic": "hydrogen_catalyst",
        "new_stance": "单原子催化产业化时间线比我预期的长，5年内不太可能大规模应用",
        "reason": "和产业界朋友交流后，了解到成本和规模化问题比论文里写的严重得多",
    })
    print(f"  结果: {result}")
    print("  ✓ PASS")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\n[TEST] give_feedback — 风格反馈")
try:
    result = agent._execute_tool("give_feedback", {
        "feedback_type": "style",
        "original_response": "这个研究有一定的学术价值。",
        "feedback": "不像我说话，我会直接说数据。这种模糊的话我不会说。",
        "context": "被问到某个催化剂工作",
    })
    print(f"  结果: {result}")
    print("  ✓ PASS")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\n[TEST] 上下文预算状态")
status = agent.context.get_budget_status()
print(f"  对话已用: {status['conversation_used']} tokens")
print(f"  动态区已用: {status['dynamic_zone']['used']} tokens")
print("  ✓ PASS")
