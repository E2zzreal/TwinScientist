# ingestion/interactive_init.py
"""Interactive initialization: CLI Q&A to fill gaps and calibrate persona."""
import os
import yaml

# Questions targeting commonly empty fields
BASE_QUESTIONS = [
    {
        "question": "你的主要研究方向是什么？（用逗号分隔多个方向）",
        "field": "research_focus",
        "type": "list",
        "target_file": "identity.yaml",
    },
    {
        "question": "用一两句话描述你的说话和思维风格？",
        "field": "personality_sketch",
        "type": "text",
        "target_file": "identity.yaml",
    },
    {
        "question": "你最确信的学术观点或信念是什么？",
        "field": "core_beliefs",
        "type": "belief",
        "target_file": "identity.yaml",
    },
    {
        "question": "你非常熟悉的领域有哪些？（用逗号分隔）",
        "field": "confident_domains",
        "type": "list",
        "target_file": "boundaries.yaml",
    },
    {
        "question": "你了解但不深入的领域？（用逗号分隔）",
        "field": "familiar_but_not_expert",
        "type": "list",
        "target_file": "boundaries.yaml",
    },
    {
        "question": "评价论文时你最先看什么？",
        "field": "review_approach",
        "type": "text",
        "target_file": "style.yaml",
    },
    {
        "question": "什么样的研究工作让你感到兴奋？",
        "field": "excited_by",
        "type": "list",
        "target_file": "identity.yaml",
    },
    {
        "question": "什么样的论文或工作让你持怀疑态度？",
        "field": "skeptical_of",
        "type": "list",
        "target_file": "identity.yaml",
    },
]


def _is_field_empty(persona_dir: str, field: str, target_file: str) -> bool:
    """Check if a persona field is empty or unset."""
    path = os.path.join(persona_dir, target_file)
    if not os.path.exists(path):
        return True
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Navigate nested fields
    if target_file == "identity.yaml":
        val = data.get(field)
        if val is None:
            taste_profile = data.get("taste_profile", {})
            if isinstance(taste_profile, dict):
                val = taste_profile.get(field)
    else:
        val = data.get(field)

    if val is None:
        return True
    if isinstance(val, (list, str)) and len(val) == 0:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def build_calibration_questions(persona_dir: str) -> list:
    """Return questions for fields that are empty in the current persona."""
    questions = []
    for q in BASE_QUESTIONS:
        if _is_field_empty(persona_dir, q["field"], q["target_file"]):
            questions.append(q)
    return questions


def _parse_answer(answer: str, q_type: str) -> object:
    """Parse raw string answer based on question type."""
    if q_type == "list":
        return [x.strip() for x in answer.split("，") + answer.split(",")
                if x.strip()]
    elif q_type == "belief":
        return [{"claim": answer.strip(), "confidence": "medium", "origin": "自述"}]
    else:
        return answer.strip()


def run_gap_filling(persona_dir: str, max_questions: int = 20) -> list:
    """Interactive CLI loop to collect answers for empty fields."""
    questions = build_calibration_questions(persona_dir)[:max_questions]
    corrections = []

    for q in questions:
        print(f"\n{q['question']}")
        print("（直接回车跳过）", end=" ")
        try:
            answer = input().strip()
        except EOFError:
            answer = ""

        if not answer:
            continue

        parsed = _parse_answer(answer, q["type"])
        corrections.append({
            "field": q["field"],
            "value": parsed,
            "target_file": q["target_file"],
        })

    return corrections


def apply_corrections(persona_dir: str, corrections: list):
    """Write user answers back into persona YAML files."""
    # Group by target file
    by_file: dict[str, list] = {}
    for c in corrections:
        by_file.setdefault(c["target_file"], []).append(c)

    for target_file, items in by_file.items():
        path = os.path.join(persona_dir, target_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        for item in items:
            field = item["field"]
            value = item["value"]

            # Handle taste_profile nested fields
            if field in ("excited_by", "skeptical_of"):
                if "taste_profile" not in data or not isinstance(data["taste_profile"], dict):
                    data["taste_profile"] = {}
                data["taste_profile"][field] = value
            else:
                data[field] = value

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def run_interactive_init(persona_dir: str):
    """Full interactive initialization flow."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        "[bold]Twin Scientist — 交互式人格初始化[/bold]\n"
        "以下问题帮助建立你的数字分身基础档案。\n"
        "直接回车可跳过任何问题。",
    ))

    corrections = run_gap_filling(persona_dir)

    if corrections:
        apply_corrections(persona_dir, corrections)
        console.print(f"\n[green]已更新 {len(corrections)} 个人格字段。[/green]")
    else:
        console.print("\n[dim]没有需要补充的内容。[/dim]")