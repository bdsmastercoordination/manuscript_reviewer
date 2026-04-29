#!/usr/bin/env python3
"""
eval/arch_critic.py

Actor-critic prompt optimizer for Mirror's archetype pipeline.

Flow per iteration
  1. Dual run  — Haiku + Sonnet independently on both stories
  2. Judge     — Sonnet scores character identification + signal-level agreement
  3. Actor     — Sonnet proposes minimal targeted prompt edit
  4. Trial     — re-run both stories with proposed prompt
  5. Accept    — only if composite score improves; write back to source

Logging
  Each dual run saves eval/annotation_logs/{story_slug}_iter{N}.json.
  Top of every log file: prompt templates used.
  Per chunk: text segment + Haiku's raw response + Sonnet's raw response.

Budget: $7.00 hard limit tracked across all API calls.
"""

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

# Force UTF-8 output so box-drawing chars survive Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import core.character_analyzer as char_mod
from core.character_analyzer import analyze_characters, CharacterSheet
from core.reviewer import _get_api_key, TokenUsage, load_docx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

HAIKU_MODEL  = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"
BUDGET       = 7.00
ITERATIONS   = 2

PRICE = {
    HAIKU_MODEL:  {"in": 0.80e-6, "out": 4.00e-6},
    SONNET_MODEL: {"in": 3.00e-6, "out": 15.00e-6},
}

STORIES = [
    (ROOT / "test_stories" / "gift_of_the_magi.docx", "Gift of the Magi"),
    (ROOT / "test_stories" / "the_necklace.docx",     "The Necklace"),
]

LOG_DIR      = ROOT / "eval" / "annotation_logs"
SUMMARY_PATH = ROOT / "eval" / "arch_critic_log.json"

ROSTER_PH     = "<<<ROSTER>>>"
ARCHETYPES_PH = "<<<ARCHETYPES>>>"
CHAR_ANALYZER  = ROOT / "core" / "character_analyzer.py"


# ═══════════════════════════════════════════════════════════════════════════════
# COST TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    budget: float = BUDGET
    spent:  float = 0.0
    input_tokens:  int = 0
    output_tokens: int = 0
    calls: int = 0

    def record(self, model: str, usage) -> float:
        p    = PRICE[model]
        # TokenUsage (from core.reviewer) uses .input/.output;
        # anthropic.types.Usage uses .input_tokens/.output_tokens
        inp  = getattr(usage, "input",  None) or getattr(usage, "input_tokens",  0)
        out  = getattr(usage, "output", None) or getattr(usage, "output_tokens", 0)
        cost = inp * p["in"] + out * p["out"]
        self.spent         += cost
        self.input_tokens  += inp
        self.output_tokens += out
        self.calls         += 1
        if self.spent > self.budget:
            raise BudgetExceeded(
                f"Budget ${self.budget:.2f} exceeded — spent ${self.spent:.2f}"
            )
        return cost

    def guard(self, headroom: float = 0.05):
        if self.remaining < headroom:
            raise BudgetExceeded(
                f"Less than ${headroom:.2f} remaining (${self.remaining:.3f})"
            )

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def line(self) -> str:
        return (
            f"  [cost] ${self.spent:.3f} / ${self.budget:.2f}  "
            f"({self.remaining:.3f} remaining, {self.calls} calls)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def banner(title: str, char: str = "═", width: int = 68):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def section(title: str):
    pad = max(0, 56 - len(title))
    print(f"\n  ── {title} {'─' * pad}")

def tick(msg: str):  print(f"  ✓ {msg}")
def info(msg: str):  print(f"  · {msg}")
def warn(msg: str):  print(f"  ⚠ {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC CLIENT  (Sonnet only — Haiku runs via analyze_characters)
# ═══════════════════════════════════════════════════════════════════════════════

_client: anthropic.Anthropic | None = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_get_api_key())
    return _client


def chat(system: str, user: str, tracker: CostTracker,
         max_tokens: int = 1024, label: str = "") -> str:
    tracker.guard()
    resp = get_client().messages.create(
        model=SONNET_MODEL,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    cost = tracker.record(SONNET_MODEL, resp.usage)
    suffix = f" → ${cost:.4f}  (total ${tracker.spent:.3f})" if label else ""
    if label:
        info(f"{label}{suffix}")
    return resp.content[0].text.strip()


def parse_json_response(raw: str) -> dict:
    """Strip optional markdown fencing and parse JSON."""
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    return json.loads(text)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT EXTRACTION / PATCHING / WRITING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_prompts() -> tuple[str, str]:
    """Return (discovery_prompt, annotation_template) from character_analyzer.py."""
    src = CHAR_ANALYZER.read_text(encoding="utf-8")

    m = re.search(r'_DISCOVERY_SYSTEM\s*=\s*"""\\\n(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("Cannot locate _DISCOVERY_SYSTEM in character_analyzer.py")
    discovery = m.group(1)

    m2 = re.search(
        r'def _annotation_system\b.*?return f"""\\\n(.*?)"""',
        src, re.DOTALL,
    )
    if not m2:
        raise RuntimeError("Cannot locate _annotation_system f-string in character_analyzer.py")
    template = (m2.group(1)
                .replace("{roster_block}",    ROSTER_PH)
                .replace("{archetype_block}", ARCHETYPES_PH))

    return discovery, template


def _make_annotation_fn(template: str):
    def _annotation_system(roster, archetypes):
        roster_block = "\n".join(
            f"  {e.name}" + (f"  (also: {', '.join(e.aliases)})" if e.aliases else "")
            for e in roster
        )
        archetype_block = "\n".join(
            f"  {a['id']}: {a['name']} — {a['description']}"
            for a in archetypes
        )
        return (template
                .replace(ROSTER_PH,     roster_block)
                .replace(ARCHETYPES_PH, archetype_block))
    return _annotation_system


def patch_prompts(discovery: str, annotation_template: str):
    char_mod._DISCOVERY_SYSTEM   = discovery
    char_mod._annotation_system  = _make_annotation_fn(annotation_template)


def write_prompts_to_file(discovery: str, annotation_template: str):
    src = CHAR_ANALYZER.read_text(encoding="utf-8")

    src = re.sub(
        r'(_DISCOVERY_SYSTEM\s*=\s*"""\\\n)(.*?)(""")',
        lambda m: m.group(1) + discovery + m.group(3),
        src, flags=re.DOTALL,
    )

    fstr_body = (annotation_template
                 .replace(ROSTER_PH,     "{roster_block}")
                 .replace(ARCHETYPES_PH, "{archetype_block}"))
    src = re.sub(
        r'(def _annotation_system\b.*?return f"""\\\n)(.*?)(""")',
        lambda m: m.group(1) + fstr_body + m.group(3),
        src, flags=re.DOTALL,
    )

    CHAR_ANALYZER.write_text(src, encoding="utf-8")
    tick("Prompts written back to character_analyzer.py")


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_annotation_log(
    story_name: str,
    iteration: int,
    discovery_prompt: str,
    annotation_template: str,
    haiku_raw: list[dict],   # [{"phase", "index", "text", "response"}]
    sonnet_raw: list[dict],
):
    """
    Write one JSON log per story per iteration.

    Structure
    ---------
    {
      "story": ...,
      "iteration": ...,
      "prompts": {
        "discovery": "...",
        "annotation_template": "..."    # <<<ROSTER>>> / <<<ARCHETYPES>>> as placeholders
      },
      "discovery_chunks": [
        { "index": 0, "text": "...", "haiku": "...", "sonnet": "..." },
        ...
      ],
      "annotation_chunks": [
        { "index": 0, "text": "...", "haiku": "...", "sonnet": "..." },
        ...
      ]
    }
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def index_by(raw: list[dict], phase: str) -> dict[int, str]:
        return {e["index"]: e["response"] for e in raw if e["phase"] == phase}

    h_disc = index_by(haiku_raw,  "discovery")
    s_disc = index_by(sonnet_raw, "discovery")
    h_ann  = index_by(haiku_raw,  "annotation")
    s_ann  = index_by(sonnet_raw, "annotation")

    all_disc_idx  = sorted(set(h_disc) | set(s_disc))
    all_ann_idx   = sorted(set(h_ann)  | set(s_ann))

    # Recover chunk texts (same for both models — deterministic split)
    disc_text = {e["index"]: e["text"] for e in haiku_raw  if e["phase"] == "discovery"}
    disc_text.update({e["index"]: e["text"] for e in sonnet_raw if e["phase"] == "discovery"})
    ann_text  = {e["index"]: e["text"] for e in haiku_raw  if e["phase"] == "annotation"}
    ann_text.update({e["index"]: e["text"] for e in sonnet_raw if e["phase"] == "annotation"})

    log = {
        "story":     story_name,
        "iteration": iteration,
        "run_at":    datetime.now().isoformat(),
        "prompts": {
            "discovery":            discovery_prompt,
            "annotation_template":  annotation_template,
        },
        "discovery_chunks": [
            {
                "index":  i,
                "text":   disc_text.get(i, ""),
                "haiku":  h_disc.get(i, ""),
                "sonnet": s_disc.get(i, ""),
            }
            for i in all_disc_idx
        ],
        "annotation_chunks": [
            {
                "index":  i,
                "text":   ann_text.get(i, ""),
                "haiku":  h_ann.get(i, ""),
                "sonnet": s_ann.get(i, ""),
            }
            for i in all_ann_idx
        ],
    }

    path = LOG_DIR / f"{_slug(story_name)}_iter{iteration}.json"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    tick(f"Annotation log → {path.name}  "
         f"({len(all_disc_idx)} discovery + {len(all_ann_idx)} annotation chunks)")


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def kendall_tau(list_a: list[str], list_b: list[str]) -> float:
    common = [x for x in list_a if x in list_b]
    if len(common) < 2:
        return 1.0 if len(common) == 1 else 0.0
    rank_b    = {x: i for i, x in enumerate(list_b)}
    positions = [rank_b[x] for x in common]
    conc = disc = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] < positions[j]:   conc += 1
            elif positions[i] > positions[j]: disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 0.0


def _match_gt(name: str, gt: list[dict]) -> Optional[str]:
    nl = name.lower().strip()
    for c in gt:
        if nl == c["name"].lower():
            return c["name"]
        if any(nl == a.lower() for a in c.get("aliases", [])):
            return c["name"]
    for c in gt:
        all_n = [c["name"]] + c.get("aliases", [])
        if any(nl in n.lower() or n.lower() in nl for n in all_n):
            return c["name"]
    return None


@dataclass
class CharMetrics:
    f1: float; precision: float; recall: float
    found: list[str]; missed: list[str]; hallucinated: list[str]


def char_metrics(sheets: list[CharacterSheet], gt: list[dict]) -> CharMetrics:
    found: set[str] = set()
    hallucinated: list[str] = []
    for s in sheets:
        m = _match_gt(s.name, gt)
        if m:   found.add(m)
        else:   hallucinated.append(s.name)
    gt_names  = {c["name"] for c in gt}
    missed    = sorted(gt_names - found)
    tp        = len(found & gt_names)
    fp        = len(hallucinated)
    fn        = len(missed)
    prec      = tp / (tp + fp) if tp + fp else 0.0
    rec       = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return CharMetrics(f1, prec, rec, sorted(found), missed, hallucinated)


@dataclass
class SignalAgreement:
    avg_tau: float
    per_character: dict   # name → tau
    disagreements: list   # [{character, tau, haiku_top3, sonnet_top3}]


def signal_agreement(
    sheets_h: list[CharacterSheet],
    sheets_s: list[CharacterSheet],
) -> SignalAgreement:
    h_by = {s.name.lower(): s for s in sheets_h}
    s_by = {s.name.lower(): s for s in sheets_s}
    taus: dict[str, float] = {}
    for hn, sh in h_by.items():
        ss = next(
            (s for sn, s in s_by.items() if hn == sn or hn in sn or sn in hn),
            None,
        )
        if ss is None:
            continue
        rank_h = [aid for aid, _, _ in sh.top_archetypes]
        rank_s = [aid for aid, _, _ in ss.top_archetypes]
        taus[sh.name] = kendall_tau(rank_h, rank_s)

    disagreements = [
        {
            "character": name,
            "tau": round(tau, 3),
            "haiku_top3":  [aid for aid, _, _ in
                            next((s.top_archetypes[:3] for s in sheets_h if s.name == name), [])],
            "sonnet_top3": [aid for aid, _, _ in
                            next((s.top_archetypes[:3] for s in sheets_s if s.name == name), [])],
        }
        for name, tau in taus.items() if tau < 0.4
    ]
    avg = sum(taus.values()) / len(taus) if taus else 0.0
    return SignalAgreement(avg, taus, disagreements)


def story_score(mh: CharMetrics, ms: CharMetrics, ag: SignalAgreement) -> float:
    return 0.5 * (mh.f1 + ms.f1) / 2 + 0.5 * ag.avg_tau


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

_GT_SYSTEM = """\
You are a literary analyst. List all significant named characters in this story.

Return JSON only — no markdown fencing:
{"characters": [
  {"name": "Canonical Full Name", "aliases": ["first name", "nickname"], "role": "brief role"}
]}

Include: characters who speak, act, decide, or have emotions shown.
Exclude: walk-ons mentioned once with no personality shown."""


def establish_ground_truth(text: str, story_name: str, tracker: CostTracker) -> list[dict]:
    section(f"Ground truth — {story_name}")
    raw  = chat(_GT_SYSTEM, f"Story:\n\n{text[:18000]}", tracker,
                max_tokens=512, label="GT Sonnet")
    data = parse_json_response(raw)
    for c in data["characters"]:
        aliases = f"  [{', '.join(c['aliases'])}]" if c.get("aliases") else ""
        info(f"  {c['name']}{aliases}  —  {c['role']}")
    return data["characters"]


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL RUN  (with raw-response capture)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    filepath: Path,
    story_name: str,
    model: str,
    tracker: CostTracker,
) -> tuple[list[CharacterSheet], list[dict]]:
    """Run analyze_characters and return (sheets, raw_entries)."""
    label    = "Haiku" if model == HAIKU_MODEL else "Sonnet"
    raw_log: list[dict] = []

    def on_raw(phase: str, idx: int, text: str, response: str):
        raw_log.append({"phase": phase, "index": idx, "text": text, "response": response})

    def on_progress(phase, current, total):
        print(f"    {label}/{story_name[:20]}: {phase} {current}/{total}   ", end="\r", flush=True)

    sheets, usage = analyze_characters(
        str(filepath), model=model, on_progress=on_progress, on_raw=on_raw,
    )
    print()   # clear \r
    cost = tracker.record(model, usage)
    inp = getattr(usage, "input", None) or getattr(usage, "input_tokens", 0)
    out = getattr(usage, "output", None) or getattr(usage, "output_tokens", 0)
    info(f"{label} · '{story_name}':  {len(sheets)} chars  "
         f"{inp}in {out}out tok  ${cost:.4f}")
    return sheets, raw_log


@dataclass
class EvalResult:
    story_name: str
    sheets_h: list[CharacterSheet]
    sheets_s: list[CharacterSheet]
    metrics_h: CharMetrics
    metrics_s: CharMetrics
    agreement: SignalAgreement
    score: float


def evaluate_all(
    stories: list[tuple],    # (path, name, gt_chars)
    tracker: CostTracker,
    iteration: int,
    disc_prompt: str,
    ann_template: str,
) -> tuple[float, list[EvalResult]]:
    results: list[EvalResult] = []

    for filepath, story_name, gt in stories:
        section(f"Dual run — {story_name}")
        sheets_h, raw_h = run_analysis(filepath, story_name, HAIKU_MODEL,  tracker)
        sheets_s, raw_s = run_analysis(filepath, story_name, SONNET_MODEL, tracker)

        # ── save annotation log ───────────────────────────────────────────────
        save_annotation_log(
            story_name, iteration,
            disc_prompt, ann_template,
            raw_h, raw_s,
        )

        mh  = char_metrics(sheets_h, gt)
        ms  = char_metrics(sheets_s, gt)
        ag  = signal_agreement(sheets_h, sheets_s)
        sc  = story_score(mh, ms, ag)

        info(f"char F1 — Haiku {mh.f1:.3f}  Sonnet {ms.f1:.3f}")
        info(f"signal τ — avg {ag.avg_tau:.3f}   story score → {sc:.3f}")
        if mh.missed or ms.missed:
            info(f"missed: {sorted(set(mh.missed + ms.missed))}")
        for d in ag.disagreements:
            info(f"disagree [{d['character']}] τ={d['tau']}  "
                 f"H:{d['haiku_top3'][:2]}  S:{d['sonnet_top3'][:2]}")

        results.append(EvalResult(story_name, sheets_h, sheets_s, mh, ms, ag, sc))

    composite = sum(r.score for r in results) / len(results)
    print(f"\n  ► Composite score: {composite:.4f}")
    print(tracker.line())
    return composite, results


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_JUDGE_CHAR = """\
You evaluate a character identification system for fiction analysis.

Given ground truth, each model's found/missed/hallucinated characters, diagnose each failure in one sentence. Then write one paragraph on the systemic discovery prompt weakness.

Return JSON only — no markdown fencing:
{
  "haiku_misses":          {"Character Name": "diagnosis"},
  "haiku_hallucinations":  {"Character Name": "diagnosis"},
  "sonnet_misses":         {"Character Name": "diagnosis"},
  "sonnet_hallucinations": {"Character Name": "diagnosis"},
  "prompt_weakness": "paragraph"
}"""

_JUDGE_SIGNAL = """\
You evaluate archetype scoring consistency between two AI models.

For each character with tau < 0.4, identify which archetype pair is confused and why.

Return JSON only — no markdown fencing:
{
  "confused_pairs": [
    {"character": "name", "arch_a": "id", "arch_b": "id", "reason": "one sentence"}
  ],
  "prompt_weakness": "paragraph on the annotation prompt's systemic issue"
}"""


def judge_chars(story_name: str, gt: list[dict],
                mh: CharMetrics, ms: CharMetrics,
                tracker: CostTracker) -> dict:
    if not (mh.missed or mh.hallucinated or ms.missed or ms.hallucinated):
        return {"haiku_misses": {}, "haiku_hallucinations": {},
                "sonnet_misses": {}, "sonnet_hallucinations": {},
                "prompt_weakness": "No character identification failures detected."}
    user = json.dumps({
        "story": story_name,
        "ground_truth": [c["name"] for c in gt],
        "haiku":  {"found": mh.found, "missed": mh.missed, "hallucinated": mh.hallucinated},
        "sonnet": {"found": ms.found, "missed": ms.missed, "hallucinated": ms.hallucinated},
    }, indent=2)
    try:
        return parse_json_response(
            chat(_JUDGE_CHAR, user, tracker, max_tokens=800,
                 label=f"judge-chars/{story_name}")
        )
    except Exception:
        return {"prompt_weakness": "Judge returned unparseable output."}


def judge_signals(story_name: str, ag: SignalAgreement, tracker: CostTracker) -> dict:
    if not ag.disagreements:
        return {"confused_pairs": [],
                "prompt_weakness": "No significant archetype disagreements detected."}
    user = json.dumps({
        "story": story_name,
        "per_character_tau": {k: round(v, 3) for k, v in ag.per_character.items()},
        "significant_disagreements": ag.disagreements,
    }, indent=2)
    try:
        return parse_json_response(
            chat(_JUDGE_SIGNAL, user, tracker, max_tokens=800,
                 label=f"judge-signals/{story_name}")
        )
    except Exception:
        return {"confused_pairs": [], "prompt_weakness": "Judge returned unparseable output."}


# ═══════════════════════════════════════════════════════════════════════════════
# ACTOR PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_ACTOR_DISC = """\
You improve a character discovery prompt for fiction analysis AI.

You receive the current prompt and specific failures with diagnoses across two stories.

Rules — follow exactly:
- Make the MINIMUM targeted edit to address only the diagnosed failures
- Do NOT restructure or reformat the prompt
- Do NOT add sections unless essential
- Do NOT change anything unrelated to the failures
- Return ONLY the complete revised prompt text — no explanation, no markdown"""

_ACTOR_ANN = """\
You improve a character archetype annotation prompt for fiction analysis AI.

The template uses <<<ROSTER>>> and <<<ARCHETYPES>>> as runtime placeholders — do NOT modify these markers.

Rules — follow exactly:
- Make the MINIMUM targeted edit to clarify scoring distinctions for the diagnosed confusion pairs
- Do NOT restructure or reformat the prompt
- Do NOT touch the <<<ROSTER>>> or <<<ARCHETYPES>>> markers
- Return ONLY the complete revised prompt template — no explanation, no markdown"""


def propose_discovery_edit(current: str, findings: list[dict], tracker: CostTracker) -> str:
    user = (f"Current discovery prompt:\n\n{current}\n\n"
            f"Failures across stories:\n\n{json.dumps(findings, indent=2)}")
    raw = chat(_ACTOR_DISC, user, tracker, max_tokens=2048, label="actor/discovery")
    # Normalise any f-string syntax back to sentinels (defensive)
    return raw.replace("{roster_block}", ROSTER_PH).replace("{archetype_block}", ARCHETYPES_PH)


def propose_annotation_edit(current: str, findings: list[dict], tracker: CostTracker) -> str:
    user = (f"Current annotation template:\n\n{current}\n\n"
            f"Archetype confusion findings:\n\n{json.dumps(findings, indent=2)}")
    raw = chat(_ACTOR_ANN, user, tracker, max_tokens=2048, label="actor/annotation")
    return raw.replace("{roster_block}", ROSTER_PH).replace("{archetype_block}", ARCHETYPES_PH)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def exec_summary(run_log: list[dict], tracker: CostTracker, changed: bool):
    banner("EXECUTIVE SUMMARY")

    scores = [(e["iteration"], e["composite"]) for e in run_log if e["phase"] == "score"]
    print("\n  Score trajectory:")
    for it, sc in scores:
        tag = "baseline" if it == 0 else f"iter {it}"
        print(f"    {tag:10s}  {sc:.4f}")
    if len(scores) >= 2:
        print(f"    {'total Δ':10s}  {scores[-1][1] - scores[0][1]:+.4f}")

    decisions = [e for e in run_log if e["phase"] == "decision"]
    if decisions:
        print("\n  Decisions:")
        for d in decisions:
            v = "✓ ACCEPTED" if d["accepted"] else "✗ REJECTED"
            print(f"    iter {d['iteration']}  {v}  Δ={d['delta']:+.4f}  target={d['target']}")

    final_iter = max((e["iteration"] for e in run_log if e["phase"] == "score"), default=0)
    char_rows  = [e for e in run_log if e["phase"] == "char" and e["iteration"] == final_iter]
    sig_rows   = [e for e in run_log if e["phase"] == "signal" and e["iteration"] == final_iter]

    if char_rows:
        print("\n  Character identification (final):")
        for r in char_rows:
            print(f"    {r['story']:25s}  Haiku F1={r['haiku_f1']:.3f}  Sonnet F1={r['sonnet_f1']:.3f}")

    if sig_rows:
        print("\n  Signal agreement (final):")
        for r in sig_rows:
            print(f"    {r['story']:25s}  avg τ={r['avg_tau']:.3f}")

    accepted = [e for e in run_log if e["phase"] == "decision" and e["accepted"]]
    if accepted:
        print(f"\n  Prompt edits accepted: {len(accepted)}")
        for a in accepted:
            print(f"    iter {a['iteration']}  →  {a['target']}")
    else:
        print("\n  No prompt edits accepted.")

    print(f"\n  {'character_analyzer.py updated.' if changed else 'character_analyzer.py unchanged.'}")

    log_files = sorted(LOG_DIR.glob("*.json")) if LOG_DIR.exists() else []
    if log_files:
        print(f"\n  Annotation logs saved ({len(log_files)} files):")
        for f in log_files:
            print(f"    {f.name}")

    print(f"\n  API cost breakdown:")
    print(f"    Total spent:   ${tracker.spent:.4f}")
    print(f"    Budget:        ${tracker.budget:.2f}")
    print(f"    Remaining:     ${tracker.remaining:.4f}")
    print(f"    API calls:     {tracker.calls}")
    print(f"    Tokens in:     {tracker.input_tokens:,}")
    print(f"    Tokens out:    {tracker.output_tokens:,}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner("ARCHETYPE ACTOR-CRITIC OPTIMIZER")
    print(f"  Budget: ${BUDGET:.2f}   Iterations: {ITERATIONS}")
    print(f"  Stories: {', '.join(n for _, n in STORIES)}")
    print(f"  Models:  Haiku (executor) + Sonnet (judge / actor)")

    tracker  = CostTracker()
    run_log: list[dict] = []
    t_start  = time.time()

    # ── extract prompts ─────────────────────────────────────────────────────
    section("Extracting current prompts")
    disc_prompt, ann_template = extract_prompts()
    tick(f"Discovery prompt:    {len(disc_prompt)} chars")
    tick(f"Annotation template: {len(ann_template)} chars")

    # ── ground truth ─────────────────────────────────────────────────────────
    banner("SETUP — GROUND TRUTH", "─")
    story_data: list[tuple] = []
    for filepath, story_name in STORIES:
        text = load_docx(str(filepath))
        gt   = establish_ground_truth(text, story_name, tracker)
        story_data.append((filepath, story_name, gt))
        run_log.append({"phase": "gt", "story": story_name,
                        "characters": [c["name"] for c in gt]})

    # ── baseline ─────────────────────────────────────────────────────────────
    banner("BASELINE  (iteration 0)", "─")
    patch_prompts(disc_prompt, ann_template)
    composite, results = evaluate_all(story_data, tracker, 0, disc_prompt, ann_template)

    run_log.append({"phase": "score", "iteration": 0, "composite": round(composite, 4)})
    for r in results:
        run_log.append({"phase": "char",   "iteration": 0, "story": r.story_name,
                        "haiku_f1": round(r.metrics_h.f1, 4),
                        "sonnet_f1": round(r.metrics_s.f1, 4)})
        run_log.append({"phase": "signal", "iteration": 0, "story": r.story_name,
                        "avg_tau": round(r.agreement.avg_tau, 4)})

    current_disc  = disc_prompt
    current_ann   = ann_template
    current_score = composite
    any_accepted  = False

    # ── iteration loop ────────────────────────────────────────────────────────
    for iteration in range(1, ITERATIONS + 1):
        banner(f"ITERATION {iteration}", "─")

        # Phase 2-3: judge
        section("Judging")
        char_findings:   list[dict] = []
        signal_findings: list[dict] = []

        for r in results:
            gt = next(gt for _, n, gt in story_data if n == r.story_name)
            cf = judge_chars(r.story_name, gt, r.metrics_h, r.metrics_s, tracker)
            sf = judge_signals(r.story_name, r.agreement, tracker)
            char_findings.append({"story": r.story_name, **cf})
            signal_findings.append({"story": r.story_name, **sf})

        # Phase 4: decide target
        avg_char   = sum((r.metrics_h.f1 + r.metrics_s.f1) / 2 for r in results) / len(results)
        avg_signal = sum(r.agreement.avg_tau for r in results) / len(results)
        info(f"avg char F1: {avg_char:.4f}   avg signal τ: {avg_signal:.4f}")

        target = "discovery" if avg_char <= avg_signal else "annotation"
        info(f"Target: {target.upper()} prompt")

        # Phase 5: actor
        section(f"Actor — proposing {target} edit")
        if target == "discovery":
            proposed_disc = propose_discovery_edit(current_disc, char_findings, tracker)
            proposed_ann  = current_ann
        else:
            proposed_ann  = propose_annotation_edit(current_ann, signal_findings, tracker)
            proposed_disc = current_disc

        delta_chars = len(proposed_disc if target == "discovery" else proposed_ann) - \
                      len(current_disc   if target == "discovery" else current_ann)
        info(f"Edit size: {delta_chars:+d} chars")

        # Phase 6: trial
        section("Trial run")
        patch_prompts(proposed_disc, proposed_ann)
        new_composite, new_results = evaluate_all(
            story_data, tracker, iteration, proposed_disc, proposed_ann,
        )
        delta     = new_composite - current_score
        old_score = current_score

        if delta > 0:
            info(f"ACCEPTED  Δ={delta:+.4f}  ({old_score:.4f} → {new_composite:.4f})")
            current_score = new_composite
            current_disc  = proposed_disc
            current_ann   = proposed_ann
            results       = new_results
            any_accepted  = True
            accepted      = True
        else:
            warn(f"REJECTED  Δ={delta:+.4f}  — reverting")
            patch_prompts(current_disc, current_ann)
            accepted = False

        run_log.append({"phase": "decision", "iteration": iteration, "target": target,
                        "old_score": round(old_score, 4), "new_score": round(new_composite, 4),
                        "delta": round(delta, 4), "accepted": accepted,
                        "delta_chars": delta_chars})
        run_log.append({"phase": "score", "iteration": iteration,
                        "composite": round(current_score, 4)})

        final_r = new_results if accepted else results
        for r in final_r:
            run_log.append({"phase": "char",   "iteration": iteration, "story": r.story_name,
                            "haiku_f1": round(r.metrics_h.f1, 4),
                            "sonnet_f1": round(r.metrics_s.f1, 4)})
            run_log.append({"phase": "signal", "iteration": iteration, "story": r.story_name,
                            "avg_tau": round(r.agreement.avg_tau, 4)})

    # ── write back accepted prompts ───────────────────────────────────────────
    if any_accepted:
        section("Writing accepted prompts to character_analyzer.py")
        write_prompts_to_file(current_disc, current_ann)
    else:
        section("No improvements — character_analyzer.py unchanged")

    # ── save run summary ──────────────────────────────────────────────────────
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_at":       datetime.now().isoformat(),
        "duration_s":   round(time.time() - t_start, 1),
        "final_score":  round(current_score, 4),
        "total_cost":   round(tracker.spent, 4),
        "entries":      run_log,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tick(f"Run summary → {SUMMARY_PATH.name}")

    exec_summary(run_log, tracker, any_accepted)


if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as e:
        print(f"\n  ⛔ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
