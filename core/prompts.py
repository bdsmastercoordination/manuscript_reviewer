import json
from pathlib import Path

ERROR_TYPES_PATH  = Path(__file__).parent.parent / "config" / "error_types.jsonl"
STYLE_CONFIG_PATH = Path(__file__).parent.parent / "config" / "style_config.json"


def _load_error_types() -> list[dict]:
    with open(ERROR_TYPES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_error_rule(i: int, entry: dict) -> str:
    lines = [f"{i}. {entry['type']} - {entry['description']}"]
    for ex in entry.get("examples") or []:
        lines.append(f"   Example: {ex}")
    if entry.get("notes"):
        lines.append(f"   Note: {entry['notes']}")
    return "\n".join(lines)


def _load_style_config() -> dict:
    return json.loads(STYLE_CONFIG_PATH.read_text(encoding="utf-8"))


def _build_style_block() -> str:
    style = _load_style_config()
    groups: dict[str, list[str]] = {}
    for cfg in style.values():
        flag_type = cfg.get("flag_as", "PUNCTUATION")
        rule = cfg["rules"][cfg["default"]]
        groups.setdefault(flag_type, []).append(f"  • {cfg['label']}: {rule}")

    all_rules: list[str] = []
    for flag_type in sorted(groups):
        all_rules.extend(groups[flag_type])
    lines = ["House style rules (apply the same TYPE classification and NONERROR threshold as for other issues):"]
    lines.extend(all_rules)
    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────

def load_error_types() -> list[dict]:
    return _load_error_types()


def load_style_config() -> dict:
    return _load_style_config()


def save_style_config(config: dict) -> None:
    STYLE_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_valid_types(enabled_types: set[str] | None = None) -> set[str]:
    all_types = {e["type"].upper() for e in _load_error_types()}
    if enabled_types is not None:
        return all_types & {t.upper() for t in enabled_types}
    return all_types


def build_system_prompt(enabled_types: set[str] | None = None) -> str:
    entries = _load_error_types()
    if enabled_types is not None:
        norm = {t.upper() for t in enabled_types}
        entries = [e for e in entries if e["type"].upper() in norm]

    type_labels = " | ".join(e["type"] for e in entries)
    rules = "\n\n".join(_build_error_rule(i + 1, e) for i, e in enumerate(entries))
    style_block = _build_style_block()
    style_section = f"\n\n{style_block}" if style_block else ""

    return f"""You are an experienced fiction editor with a sharp eye for detail.
Review the provided manuscript excerpt and identify only these {len(entries)} types of issues:

{rules}{style_section}

Output structured entries only — no preamble, no reasoning, no category headings.
For each potential issue found, use this exact format:
TEXT: "[quote the exact problematic text, up to ~10 words]"
ISSUE: [one sentence describing the potential issue, or why on reflection this is not a genuine error]
TYPE: [{type_labels} | NONERROR]
---

Use TYPE: NONERROR when on reflection the text does not meet the threshold for a genuine issue.
If no issues are found in the excerpt, respond with exactly: NO ISSUES FOUND

Do not flag intentional stylistic choices or dialect in dialogue."""


def build_review_prompt(text: str, enabled_types: set[str] | None = None) -> str:
    entries = _load_error_types()
    if enabled_types is not None:
        norm = {t.upper() for t in enabled_types}
        entries = [e for e in entries if e["type"].upper() in norm]

    categories = "\n".join(
        f"{i + 1}. {e['type'].capitalize()} errors" for i, e in enumerate(entries)
    )
    return f"""Review this manuscript excerpt. Work through each category in order:
{categories}

---
{text}
---"""
