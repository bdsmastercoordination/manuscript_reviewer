"""
Agent-critic loop for automated prompt improvement.

The judge (Sonnet) reads each document chunk alongside Haiku's flags and returns:
  TP  — flag is a genuine editorial issue
  FP  — flag is not a real issue in context
  FN  — a genuine issue Haiku missed (identified by the judge from scratch)

  score = 0.7 * recall  -  0.3 * fp_rate
        = 0.7 * TP/(TP+FN)  -  0.3 * FP/(TP+FP)

Each iteration: find the worst-performing error type, ask Sonnet to rewrite its
definition, re-run Haiku, accept the change only if the score improves.

Run: python eval/agent_critic.py [--max-iter N]
"""

import sys, io, json, re, argparse, time
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from core.reviewer import review_document, load_docx, chunk_text, ChunkResult, Flag, _get_api_key

# ── paths & weights ───────────────────────────────────────────────────────────
LONGSTORY_PATH   = ROOT / "eval" / "longstory.docx"
ERROR_TYPES_PATH = ROOT / "config" / "error_types.jsonl"
LOG_PATH         = ROOT / "eval" / "agent_critic_log.json"

W_RECALL = 0.7
W_FP     = 0.3

JUDGE_MODEL   = "claude-sonnet-4-6"
PROPOSE_MODEL = "claude-sonnet-4-6"
PATIENCE      = 3


# ── error_types.jsonl helpers ─────────────────────────────────────────────────

def load_error_types() -> list[dict]:
    with open(ERROR_TYPES_PATH, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]

def save_error_types(entries: list[dict]) -> None:
    with open(ERROR_TYPES_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def get_entry(etype: str) -> dict | None:
    for e in load_error_types():
        if e["type"].upper() == etype.upper():
            return deepcopy(e)
    return None

def set_entry(etype: str, new_def: dict) -> None:
    entries = load_error_types()
    for i, e in enumerate(entries):
        if e["type"].upper() == etype.upper():
            entries[i] = new_def
            save_error_types(entries)
            return

def _type_list_compact() -> str:
    lines = []
    for e in load_error_types():
        first = e.get("description", "").split(".")[0]
        lines.append(f"  - {e['type']}: {first}")
    return "\n".join(lines)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    flag_idx: int
    verdict:  str    # "TP" or "FP"
    reason:   str


@dataclass
class ChunkJudgment:
    chunk_idx: int
    flags:     list[Flag]
    verdicts:  list[Verdict]
    missed:    list[dict]    # {"type", "text", "reason"}

    @property
    def tp_ids(self) -> set[int]:
        return {v.flag_idx for v in self.verdicts if v.verdict == "TP"}

    @property
    def fp_ids(self) -> set[int]:
        return {v.flag_idx for v in self.verdicts if v.verdict == "FP"}


@dataclass
class TypeStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else None

    @property
    def score(self) -> float:
        r = self.recall    if self.recall    is not None else 1.0
        p = self.precision if self.precision is not None else 1.0
        return W_RECALL * r - W_FP * (1 - p)

    def badness(self) -> float:
        r = self.recall    if self.recall    is not None else 0.5
        p = self.precision if self.precision is not None else 0.5
        return W_RECALL * (1 - r) + W_FP * (1 - p)


@dataclass
class IterLog:
    iteration:  int
    score:      float
    recall:     float
    precision:  float
    tp: int; fp: int; fn: int
    target:     str | None
    accepted:   bool
    delta:      float
    type_stats: dict[str, TypeStats]


# ── judge ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a senior fiction editor evaluating Mirror's output — an AI fiction editor. "
    "Be precise and critical — flag false positives aggressively but do not invent problems."
)

def _judge_prompt(chunk_text: str, flags: list[Flag]) -> str:
    type_list = _type_list_compact()
    flag_lines = "\n".join(
        f'  {i}. [{f.type}] "{f.text}" — {f.issue}'
        for i, f in enumerate(flags)
    ) or "  (none — model found no issues)"

    return f"""Evaluate this AI fiction editor's output on the passage below.

PASSAGE:
\"\"\"
{chunk_text.strip()}
\"\"\"

FLAGS RAISED:
{flag_lines}

AVAILABLE ERROR TYPES (use these names exactly for missed issues):
{type_list}

Instructions:
1. For each flag, decide TP (genuine editorial issue a professional would mark) or FP (not a real problem in this context).
2. List any genuine issues the AI missed — only from the available types above.
   Be conservative: only list issues you are confident about.

Respond ONLY with valid JSON, no markdown fences:
{{
  "verdicts": [
    {{"id": 0, "verdict": "TP", "reason": "one sentence"}},
    {{"id": 1, "verdict": "FP", "reason": "one sentence"}}
  ],
  "missed": [
    {{"type": "EXACT TYPE NAME", "text": "quoted text from passage", "reason": "one sentence"}}
  ]
}}"""


def _parse_judge(raw: str, n_flags: int) -> tuple[list[Verdict], list[dict]]:
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("    [judge] JSON parse failed — treating all flags as TP")
        return [Verdict(i, "TP", "parse error") for i in range(n_flags)], []

    verdicts = []
    seen_ids = set()
    for v in data.get("verdicts", []):
        fid = int(v.get("id", -1))
        ver = str(v.get("verdict", "TP")).upper()
        if 0 <= fid < n_flags and fid not in seen_ids:
            verdicts.append(Verdict(fid, ver if ver in ("TP", "FP") else "TP", v.get("reason", "")))
            seen_ids.add(fid)

    # Any flagged item not in verdicts → default TP
    for i in range(n_flags):
        if i not in seen_ids:
            verdicts.append(Verdict(i, "TP", "not evaluated"))

    missed = []
    valid_types = {e["type"].upper() for e in load_error_types()}
    for m in data.get("missed", []):
        etype = str(m.get("type", "")).upper().strip()
        if etype in valid_types:
            missed.append({"type": etype, "text": m.get("text", ""), "reason": m.get("reason", "")})

    return verdicts, missed


def judge_chunk(client: anthropic.Anthropic, chunk_text: str,
                chunk_idx: int, flags: list[Flag]) -> ChunkJudgment:
    prompt = _judge_prompt(chunk_text, flags)
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1200,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    verdicts, missed = _parse_judge(resp.content[0].text, len(flags))
    return ChunkJudgment(chunk_idx=chunk_idx, flags=flags,
                         verdicts=verdicts, missed=missed)


# ── scoring ───────────────────────────────────────────────────────────────────

def compute_stats(
    judgments: list[ChunkJudgment],
) -> tuple[float, float, float, int, int, int, dict[str, TypeStats]]:
    by_type: dict[str, TypeStats] = defaultdict(TypeStats)
    total_tp = total_fp = total_fn = 0

    for j in judgments:
        tp_ids = j.tp_ids
        for i, f in enumerate(j.flags):
            etype = f.type.upper()
            if i in tp_ids:
                by_type[etype].tp += 1
                total_tp += 1
            else:
                by_type[etype].fp += 1
                total_fp += 1
        for m in j.missed:
            etype = m["type"].upper()
            by_type[etype].fn += 1
            total_fn += 1

    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    score     = W_RECALL * recall - W_FP * (1 - precision)
    return score, recall, precision, total_tp, total_fp, total_fn, dict(by_type)


# ── proposer ──────────────────────────────────────────────────────────────────

_PROPOSE_SYSTEM = (
    "You are a prompt engineer improving an AI fiction editor's error detection rules. "
    "Output only the requested JSON — no explanation, no markdown."
)

def _propose_prompt(etype: str, current: dict,
                    stats: TypeStats,
                    fp_examples: list[dict],
                    fn_examples: list[dict]) -> str:
    fp_block = "\n".join(
        f'  - "{e["text"]}": {e["reason"]}' for e in fp_examples[:4]
    ) or "  (none)"
    fn_block = "\n".join(
        f'  - "{e["text"]}": {e["reason"]}' for e in fn_examples[:4]
    ) or "  (none)"
    rec  = f"{stats.recall:.0%} ({stats.tp} caught, {stats.fn} missed)" if stats.recall is not None else f"N/A ({stats.fn} missed)"
    prec = f"{stats.precision:.0%} ({stats.tp} correct, {stats.fp} false positives)" if stats.precision is not None else f"N/A ({stats.fp} false positives)"

    return f"""Improve this fiction editor error type definition.

TYPE: {etype}

CURRENT DEFINITION:
{json.dumps(current, indent=2, ensure_ascii=False)}

JUDGE-EVALUATED PERFORMANCE:
  Recall:    {rec}
  Precision: {prec}

FALSE POSITIVES (flagged but judge said NOT a genuine issue):
{fp_block}

MISSED (genuine issues the model did not catch):
{fn_block}

Rewrite the definition to:
  1. Reduce false positives — add specific exclusion criteria matching the FP patterns above
  2. Improve recall — sharpen examples to match the missed patterns above
  3. Keep the same overall intent of the error type

Output ONLY a JSON object with exactly these fields:
  "type"        : "{etype}"  (do not change)
  "description" : revised description string
  "examples"    : list of example strings
  "notes"       : specific exclusion criteria and edge cases"""


def propose_revision(client: anthropic.Anthropic, etype: str,
                     stats: TypeStats,
                     fp_examples: list[dict],
                     fn_examples: list[dict]) -> dict | None:
    current = get_entry(etype)
    if current is None:
        return None
    prompt = _propose_prompt(etype, current, stats, fp_examples, fn_examples)
    resp = client.messages.create(
        model=PROPOSE_MODEL,
        max_tokens=900,
        system=_PROPOSE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        new_def = json.loads(raw)
        assert new_def.get("type", "").upper() == etype.upper()
        assert "description" in new_def
        return new_def
    except Exception as e:
        print(f"    [proposer] parse failed: {e}")
        return None


# ── target selection ──────────────────────────────────────────────────────────

def find_worst_type(type_stats: dict[str, TypeStats],
                    recent_targets: list[str]) -> str | None:
    skip = set(recent_targets[-2:])
    candidates = [
        (stats.badness(), etype)
        for etype, stats in type_stats.items()
        if etype not in skip and (stats.tp + stats.fp + stats.fn) > 0
    ]
    if not candidates:
        candidates = [
            (stats.badness(), etype)
            for etype, stats in type_stats.items()
            if (stats.tp + stats.fp + stats.fn) > 0
        ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def collect_examples(judgments: list[ChunkJudgment],
                     target: str) -> tuple[list[dict], list[dict]]:
    fp_examples, fn_examples = [], []
    for j in judgments:
        fp_ids = j.fp_ids
        for v in j.verdicts:
            if v.verdict == "FP" and j.flags[v.flag_idx].type.upper() == target:
                fp_examples.append({"text": j.flags[v.flag_idx].text, "reason": v.reason})
        for m in j.missed:
            if m["type"].upper() == target:
                fn_examples.append(m)
    return fp_examples, fn_examples


# ── main loop ─────────────────────────────────────────────────────────────────

def run_full_evaluation(client: anthropic.Anthropic,
                        chunks: list[str]) -> tuple[float, float, float, int, int, int,
                                                    dict[str, TypeStats], list[ChunkJudgment]]:
    print("  Running Haiku reviewer...", flush=True)
    results, usage = review_document(str(LONGSTORY_PATH))
    flags_by_chunk: dict[int, list[Flag]] = defaultdict(list)
    for r in results:
        flags_by_chunk[r.chunk - 1].extend(r.flags)

    print(f"  Haiku: {sum(len(f) for f in flags_by_chunk.values())} flags across "
          f"{len(results)} chunks  [{usage.input:,}in/{usage.output:,}out]", flush=True)

    print("  Judging all chunks with Sonnet...", flush=True)
    judgments = []
    for idx, chunk in enumerate(chunks):
        flags = flags_by_chunk.get(idx, [])
        j = judge_chunk(client, chunk, idx, flags)
        tp_n = len(j.tp_ids)
        fp_n = len(j.fp_ids)
        fn_n = len(j.missed)
        print(f"    chunk {idx+1}/{len(chunks)}: "
              f"{len(flags)} flags → {tp_n} TP / {fp_n} FP / {fn_n} FN missed",
              flush=True)
        judgments.append(j)

    score, recall, prec, tp, fp, fn, type_stats = compute_stats(judgments)
    return score, recall, prec, tp, fp, fn, type_stats, judgments


def run(max_iter: int) -> None:
    client = anthropic.Anthropic(api_key=_get_api_key())

    print("Loading document and chunking...", flush=True)
    raw = load_docx(str(LONGSTORY_PATH))
    chunks = chunk_text(raw)
    print(f"  {len(chunks)} chunks from longstory.docx\n", flush=True)

    history:          list[IterLog]         = []
    recent_targets:   list[str]             = []
    accepted_changes: list[tuple[str, float, dict, dict]] = []  # (type, delta, old, new)
    no_improve       = 0
    best_score       = -999.0

    # ── baseline ──────────────────────────────────────────────────────────────
    print("=" * 66)
    print("BASELINE (iteration 0)")
    print("=" * 66)
    score, recall, prec, tp, fp, fn, type_stats, base_judgments = run_full_evaluation(client, chunks)
    best_score = score
    history.append(IterLog(
        iteration=0, score=score, recall=recall, precision=prec,
        tp=tp, fp=fp, fn=fn,
        target=None, accepted=True, delta=0.0,
        type_stats=deepcopy(type_stats),
    ))
    print(f"\n  Score={score:.4f}  Recall={recall:.3f}  Precision={prec:.3f}  "
          f"TP={tp}  FP={fp}  FN={fn}\n")
    baseline_type_stats = deepcopy(type_stats)

    # Populate cache from baseline so first proposer call has examples
    for etype in type_stats:
        ex_fp, ex_fn = collect_examples(base_judgments, etype)
        _fp_fn_cache[etype] = (ex_fp, ex_fn)

    _save_log(history)

    # ── iterations ────────────────────────────────────────────────────────────
    for it in range(1, max_iter + 1):
        print("=" * 66)
        print(f"ITERATION {it}/{max_iter}")
        print("=" * 66)

        target = find_worst_type(type_stats, recent_targets)
        if target is None:
            print("No viable target type found — stopping.")
            break
        recent_targets.append(target)
        t_stats = type_stats.get(target, TypeStats())
        rec_str  = f"{t_stats.recall:.0%}"    if t_stats.recall    is not None else "N/A"
        prec_str = f"{t_stats.precision:.0%}" if t_stats.precision is not None else "N/A"
        print(f"  Target: {target}  "
              f"(recall={rec_str} prec={prec_str} "
              f"TP={t_stats.tp} FP={t_stats.fp} FN={t_stats.fn})",
              flush=True)

        fp_ex, fn_ex = _fp_fn_cache.get(target, ([], []))

        print(f"  Examples: {len(fp_ex)} FP, {len(fn_ex)} FN", flush=True)
        old_def = get_entry(target)

        print("  Proposing revision...", flush=True)
        new_def = propose_revision(client, target, t_stats, fp_ex, fn_ex)
        if new_def is None:
            print("  Proposer failed — skipping iteration.\n")
            continue

        set_entry(target, new_def)
        print("  Re-evaluating...", flush=True)
        new_score, new_recall, new_prec, new_tp, new_fp, new_fn, new_type_stats, new_judgments = \
            run_full_evaluation(client, chunks)

        delta    = new_score - best_score
        accepted = new_score > best_score

        if accepted:
            best_score    = new_score
            type_stats    = new_type_stats
            no_improve    = 0
            accepted_changes.append((target, delta, old_def, new_def))
            _fp_fn_cache.clear()
            touched = set()
            for j in new_judgments:
                touched |= {j.flags[i].type.upper() for i in j.fp_ids}
                touched |= {m["type"].upper() for m in j.missed}
            for etype in touched:
                ex_fp, ex_fn = collect_examples(new_judgments, etype)
                _fp_fn_cache[etype] = (ex_fp, ex_fn)
            print(f"\n  ✓ ACCEPTED  score {best_score:.4f} ({delta:+.4f})")
        else:
            set_entry(target, old_def)   # rollback
            type_stats = type_stats      # keep old stats
            no_improve += 1
            print(f"\n  ✗ REJECTED  score {new_score:.4f} ({delta:+.4f})  "
                  f"[rollback — no_improve={no_improve}/{PATIENCE}]")

        history.append(IterLog(
            iteration=it, score=new_score if accepted else history[-1].score,
            recall=new_recall if accepted else history[-1].recall,
            precision=new_prec if accepted else history[-1].precision,
            tp=new_tp if accepted else history[-1].tp,
            fp=new_fp if accepted else history[-1].fp,
            fn=new_fn if accepted else history[-1].fn,
            target=target, accepted=accepted, delta=delta,
            type_stats=deepcopy(new_type_stats if accepted else type_stats),
        ))
        print()
        _save_log(history)

        if no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} consecutive iterations — stopping early.\n")
            break

    _print_report(history, baseline_type_stats, accepted_changes)


# ── helper for FP/FN cache ────────────────────────────────────────────────────
_fp_fn_cache: dict[str, tuple[list, list]] = {}


def _last_judgments(history: list[IterLog]):
    pass  # placeholder; cache is populated during the loop


# ── log ───────────────────────────────────────────────────────────────────────

def _save_log(history: list[IterLog]) -> None:
    data = []
    for h in history:
        data.append({
            "iteration": h.iteration, "score": h.score,
            "recall": h.recall, "precision": h.precision,
            "tp": h.tp, "fp": h.fp, "fn": h.fn,
            "target": h.target, "accepted": h.accepted, "delta": h.delta,
        })
    LOG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── report ────────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    return f"{v:.0%}" if v is not None else "  N/A"

def _print_report(history: list[IterLog],
                  baseline: dict[str, TypeStats],
                  accepted_changes: list[tuple]) -> None:
    div = "=" * 72
    print(f"\n{div}")
    print("AGENT-CRITIC REPORT")
    print(div)

    # ── learning curve ────────────────────────────────────────────────────────
    print("\nLEARNING CURVE")
    print(f"  {'Iter':>4}  {'Score':>7}  {'Recall':>7}  {'Precis':>7}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'Target':<24}  {'Acc':>3}  {'Δ':>7}")
    print("  " + "-" * 68)
    for h in history:
        acc   = " ✓" if h.accepted else " ✗"
        delta = f"{h.delta:+.4f}" if h.iteration > 0 else "      —"
        tgt   = (h.target or "(baseline)")[:24]
        print(f"  {h.iteration:>4}  {h.score:>7.4f}  {h.recall:>7.3f}  "
              f"{h.precision:>7.3f}  {h.tp:>4}  {h.fp:>4}  {h.fn:>4}  "
              f"{tgt:<24}  {acc:>3}  {delta:>7}")

    # ── per-type summary ──────────────────────────────────────────────────────
    final = history[-1].type_stats
    all_types = sorted(set(baseline) | set(final))

    print(f"\n\nPER-TYPE SUMMARY")
    print(f"  {'Type':<26}  {'—— Baseline ——':^30}  {'—— Final ——':^30}  {'Δ Prec':>7}  {'Δ Rec':>6}")
    print(f"  {'':26}  {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6}  "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6}  {'':>7}  {'':>6}")
    print("  " + "-" * 96)

    for etype in all_types:
        b = baseline.get(etype, TypeStats())
        f = final.get(etype, TypeStats())
        d_prec = (f.precision or 0) - (b.precision or 0)
        d_rec  = (f.recall    or 0) - (b.recall    or 0)
        sign_p = "+" if d_prec >= 0 else ""
        sign_r = "+" if d_rec  >= 0 else ""
        print(f"  {etype:<26}  "
              f"{b.tp:>4} {b.fp:>4} {b.fn:>4} {_pct(b.precision):>6} {_pct(b.recall):>6}  "
              f"{f.tp:>4} {f.fp:>4} {f.fn:>4} {_pct(f.precision):>6} {_pct(f.recall):>6}  "
              f"{sign_p}{d_prec:>6.0%}  {sign_r}{d_rec:>5.0%}")

    # ── most impactful changes ────────────────────────────────────────────────
    print(f"\n\nMOST IMPACTFUL ACCEPTED CHANGES")
    if not accepted_changes:
        print("  No changes were accepted.")
    else:
        ranked = sorted(accepted_changes, key=lambda x: x[1], reverse=True)
        for rank, (etype, delta, old_def, new_def) in enumerate(ranked, 1):
            print(f"\n  {rank}. {etype}  ({delta:+.4f})")
            old_desc = old_def.get("description", "")[:120]
            new_desc = new_def.get("description", "")[:120]
            old_notes = old_def.get("notes", "")[:100]
            new_notes = new_def.get("notes", "")[:100]
            if old_desc != new_desc:
                print(f"     description:")
                print(f"       was: {old_desc}")
                print(f"       now: {new_desc}")
            if old_notes != new_notes:
                print(f"     notes:")
                print(f"       was: {old_notes}")
                print(f"       now: {new_notes}")

    # ── flagged types ─────────────────────────────────────────────────────────
    print(f"\n\nSTILL UNDERPERFORMING (precision < 50% in final iteration)")
    poor = [(e, s) for e, s in final.items()
            if s.precision is not None and s.precision < 0.5]
    if poor:
        for etype, s in sorted(poor, key=lambda x: x[1].precision or 0):
            print(f"  {etype:<26}  prec={_pct(s.precision)}  "
                  f"TP={s.tp}  FP={s.fp}  FN={s.fn}")
    else:
        print("  None — all types with data are at ≥50% precision.")

    # ── overall summary ───────────────────────────────────────────────────────
    b0 = history[0]
    bf = history[-1]
    print(f"\n\nOVERALL")
    print(f"  Baseline score : {b0.score:.4f}  "
          f"(recall={b0.recall:.3f}  precision={b0.precision:.3f}  "
          f"TP={b0.tp}  FP={b0.fp}  FN={b0.fn})")
    print(f"  Final score    : {bf.score:.4f}  "
          f"(recall={bf.recall:.3f}  precision={bf.precision:.3f}  "
          f"TP={bf.tp}  FP={bf.fp}  FN={bf.fn})")
    print(f"  Δ score        : {bf.score - b0.score:+.4f}")
    print(f"  Accepted edits : {len(accepted_changes)} / {len(history)-1} iterations")
    print(f"\n  Log saved to   : {LOG_PATH}")
    print(div + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=12)
    args = parser.parse_args()

    if not LONGSTORY_PATH.exists():
        sys.exit(f"longstory.docx not found at {LONGSTORY_PATH}")

    t0 = time.time()
    run(args.max_iter)
    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed/60:.1f} min")
