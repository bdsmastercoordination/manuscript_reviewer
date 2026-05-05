"""
Two-pass character analysis for Mirror.

Pass 1  large chunks (~8 000 tokens)
        Asks the model who the main characters are; extracts canonical name,
        aliases, and a 2-sentence synopsis for each.  Runs once per large
        chunk, then a roster of the top-K characters is assembled in code.

Pass 2  small chunks (~1 000 tokens)
        For each chunk, asks the model to find moments where a roster
        character acts, speaks, or thinks, and score the moment against all
        50 archetypes (0–10).  Only scores ≥ 3 are recorded.

Aggregation
        For each character, archetype scores are summed across every signal
        instance and divided by the total number of signal instances (not
        just the instances where that archetype fired).  This rewards
        consistent presence over isolated intensity.
"""

import json
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx

# ── constants ─────────────────────────────────────────────────────────────────

ARCHETYPES_PATH        = Path(__file__).parent.parent / "config" / "archetypes.jsonl"
DISCOVERY_CHUNK_CHARS  = 24_000   # ≈ 8 000 tokens
ANNOTATION_CHUNK_CHARS = 3_500    # ≈ 1 000 tokens
DEFAULT_ROSTER_SIZE    = 3
MAX_CONCURRENT         = 3
DEFAULT_MODEL          = "claude-haiku-4-5-20251001"


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk(text: str, target_chars: int) -> list[str]:
    """Split text into chunks at paragraph boundaries, aiming for target_chars."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > target_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:target_chars]]


def load_archetypes() -> list[dict]:
    with open(ARCHETYPES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class RosterEntry:
    name:        str
    aliases:     list[str]
    synopsis:    str
    chunks_seen: int   = 0
    avg_rank:    float = 0.0


@dataclass
class _Signal:
    """One annotated moment from Pass 2."""
    character: str
    signal:    str
    scores:    dict[str, float]   # archetype_id → score (only those ≥ 3)
    chunk_idx: int = 0
    offset:    int = 50           # 0–100, position within the excerpt


@dataclass
class CharacterSheet:
    name:            str
    aliases:         list[str]
    synopsis:        str
    top_archetypes:  list[tuple[str, str, float]]  # (id, display_name, avg_score)
    evidence:        dict[str, list[str]]           # archetype_id → [signal quotes]
    total_signals:   int
    chunks_seen:     int
    signals_ordered:     list = field(default_factory=list)
    neg_signals_ordered: list = field(default_factory=list)
    # each entry: {"chunk_idx": int, "scores": dict[str, float]}


# ── Pass 1: prompts and parsing ───────────────────────────────────────────────

_DISCOVERY_SYSTEM = """\
You identify and rank characters in fiction manuscripts.

Given an excerpt, list every character who acts, speaks, thinks, or is \
described with enough detail to reveal personality. Include protagonists, \
supporting characters, and secondary figures alike. Exclude only pure \
walk-ons — characters who are merely named in passing with no behaviour, \
dialogue, or inner life shown.

In first-person or close-third narration the narrator is often a character \
themselves — include them if they act, feel, or reveal personality, even if \
unnamed (use "Narrator" as the name in that case).

List characters in order of centrality in this specific excerpt — the \
character who most drives, dominates, or anchors the scene comes first. \
Be inclusive: err toward listing more characters rather than fewer; a \
later step handles the final cutdown.

Rules:
- One CHARACTER block per individual. Never combine multiple people into \
a single block. If group members are unnamed, skip them.
- The CHARACTER field contains exactly one person's name as written in the text.

For each character output exactly:

CHARACTER: [name of one individual, as written in the text]
SYNOPSIS: [exactly two sentences: approximate age and occupation if inferable; their role in the story]
---

If no characters are clearly present in this excerpt, output: NO CHARACTERS
No text outside the structured blocks."""

_RECONCILE_SYSTEM = """\
You are given a list of character names collected from different passages of \
a fiction manuscript. The same character may appear under different names \
across passages (e.g. "Gregor" and "Gregor Samsa", or "Grete" and \
"Gregor's sister", or "the charwoman" and "the cleaner").

Your task: group all names that refer to the exact same individual. \
For each group, choose the most complete and descriptive name as the \
canonical name and list all other names as aliases.

Rules:
- Only group names that unambiguously refer to the same person. When in \
doubt, keep them separate.
- Do not group names just because characters are related (e.g. \
"Gregor's mother" and "Gregor's father" are two different people).
- A name that clearly belongs to only one character should appear in \
exactly one group.
- Use the story synopsis if provided to resolve ambiguous or \
role-based names (e.g. "the stranger" or "the old woman").

Output one block per individual character:

CANONICAL: [the most complete or descriptive name for this person]
ALIASES: [comma-separated other names for this same person, or "none"]
---

No text outside the structured blocks."""


def _discovery_prompt(chunk: str) -> str:
    return f"Who are the important characters in this excerpt?\n\n{chunk}"


def _parse_discovery(response: str) -> list[RosterEntry]:
    if "NO CHARACTERS" in response.upper():
        return []
    entries: list[RosterEntry] = []
    for block in re.split(r"\n---+\n?", response):
        block = block.strip()
        if not block:
            continue
        name_m = re.search(r"^CHARACTER:\s*(.+)$", block, re.MULTILINE)
        syn_m  = re.search(r"^SYNOPSIS:\s*(.+)$",  block, re.MULTILINE | re.DOTALL)
        if not name_m:
            continue
        synopsis = ""
        if syn_m:
            synopsis = re.split(r"\n[A-Z ]+:", syn_m.group(1))[0].strip()
        entries.append(RosterEntry(
            name=name_m.group(1).strip(),
            aliases=[],
            synopsis=synopsis,
        ))
    return entries


def _parse_reconciliation(response: str) -> list[RosterEntry]:
    entries: list[RosterEntry] = []
    for block in re.split(r"\n---+\n?", response):
        block = block.strip()
        if not block:
            continue
        canon_m = re.search(r"^CANONICAL:\s*(.+)$", block, re.MULTILINE)
        alias_m = re.search(r"^ALIASES:\s*(.+)$",   block, re.MULTILINE)
        if not canon_m:
            continue
        raw = alias_m.group(1).strip() if alias_m else ""
        aliases = (
            []
            if raw.lower() in ("none", "")
            else [a.strip().strip('"\'') for a in raw.split(",") if a.strip()]
        )
        entries.append(RosterEntry(name=canon_m.group(1).strip(), aliases=aliases, synopsis=""))
    return entries


# ── roster aggregation ────────────────────────────────────────────────────────

def _first_word(name: str) -> str:
    return name.strip().lower().split()[0] if name.strip() else ""


async def _aggregate_roster(
    per_chunk: list[list[RosterEntry]],
    k: int,
    client,
    model: str,
    synopsis: str = "",
) -> list[RosterEntry]:
    """
    Two-step aggregation:
    1. Collect raw names and their chunk-position stats (no heuristic merging).
    2. Single LLM call to reconcile which names refer to the same character.
    """
    # Step 1: accumulate raw names with appearance stats
    # raw_name → {chunks: [chunk_idx, ...], ranks: [position_in_chunk, ...], synopsis: str}
    raw_stats: dict[str, dict] = {}
    for chunk_idx, entries in enumerate(per_chunk):
        for rank, entry in enumerate(entries, start=1):
            name = entry.name.strip()
            if name not in raw_stats:
                raw_stats[name] = {"chunks": [], "ranks": [], "synopsis": entry.synopsis}
            raw_stats[name]["chunks"].append(chunk_idx)
            raw_stats[name]["ranks"].append(rank)
            if len(entry.synopsis) > len(raw_stats[name]["synopsis"]):
                raw_stats[name]["synopsis"] = entry.synopsis

    if not raw_stats:
        return []

    # Step 2: LLM reconciliation
    name_list = "\n".join(f"- {n}" for n in raw_stats)
    if not synopsis.strip():
        raise ValueError("Synopsis is required for character reconciliation but was not provided.")
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        system=_RECONCILE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Story synopsis:\n{synopsis}\n\nReconcile these character names:\n\n{name_list}"}],
    )
    groups = _parse_reconciliation(resp.content[0].text.strip())

    # Step 3: merge stats through canonical groups and rank
    result: list[RosterEntry] = []
    claimed: set[str] = set()

    for group in groups:
        all_names = [group.name] + group.aliases
        # gather stats from all names in this group
        chunks_seen: set[int] = set()
        rank_sum = 0.0
        rank_count = 0
        best_synopsis = ""
        for n in all_names:
            if n in raw_stats:
                claimed.add(n)
                s = raw_stats[n]
                chunks_seen.update(s["chunks"])
                rank_sum   += sum(s["ranks"])
                rank_count += len(s["ranks"])
                if len(s["synopsis"]) > len(best_synopsis):
                    best_synopsis = s["synopsis"]
        if rank_count == 0:
            continue
        avg_rank = rank_sum / rank_count
        entry = RosterEntry(
            name=group.name,
            aliases=group.aliases,
            synopsis=best_synopsis,
            chunks_seen=len(chunks_seen),
        )
        entry.avg_rank = avg_rank
        result.append(entry)

    # Any raw name the LLM didn't place in a group gets its own entry
    for name, s in raw_stats.items():
        if name not in claimed:
            avg_rank = sum(s["ranks"]) / len(s["ranks"])
            entry = RosterEntry(
                name=name, aliases=[], synopsis=s["synopsis"],
                chunks_seen=len(set(s["chunks"])),
            )
            entry.avg_rank = avg_rank
            result.append(entry)

    result.sort(key=lambda e: e.avg_rank / max(e.chunks_seen, 1))
    return result[:k]


# ── Pass 2: prompts and parsing ───────────────────────────────────────────────

def _annotation_system(roster: list[RosterEntry], archetypes: list[dict]) -> str:
    roster_block = "\n".join(
        f"  {e.name}" + (f"  (also: {', '.join(e.aliases)})" if e.aliases else "")
        for e in roster
    )
    archetype_block = "\n".join(
        f"  {a['id']}: {a['name']} — {a['description']}"
        for a in archetypes
    )
    return f"""\
You annotate fiction manuscripts for character behaviour and archetypes.

Characters to watch for:
{roster_block}

Read the excerpt and find every moment where one of these characters acts, \
speaks, or (in close third / first person) thinks — including small, quiet, \
or ambiguous moments, not only dramatic ones.

For each moment, work in two steps:
1. Describe what the character does, says, or thinks in your own words \
(their behaviour, attitude, or inner state).
2. Score how strongly each archetype below fits that specific behaviour \
(0 = does not apply, 10 = defines this archetype perfectly). \
List only archetypes scoring 3 or higher; if none reach 3, still output \
the block with ARCHETYPES: none.

When two archetypes share a surface trait (e.g. selfless devotion), use the \
character's apparent motivation as the tiebreaker: ask whether the behaviour \
is driven primarily by an internal vision or ideal (score the idealist-type \
higher) versus by loyalty and a desire to please or serve the other person \
(score the devoted-companion-type higher). Apply this motivational test \
before finalising scores for any archetypes whose descriptions overlap.

Archetypes:
{archetype_block}

Output one block per moment:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence quote or tight paraphrase of the moment]
BEHAVIOR: [your one-sentence reflection on what this reveals about the character]
ARCHETYPES: [id:score, id:score, ...] or "none"
OFFSET: [0-100, where 0 = start of excerpt, 100 = end]
---

If no listed characters appear in this excerpt at all, output: NO SIGNALS
No text outside the structured blocks."""


def _annotation_prompt(chunk: str) -> str:
    return f"Annotate this excerpt:\n\n{chunk}"


def _parse_signals(response: str, roster: list[RosterEntry]) -> list[_Signal]:
    if "NO SIGNALS" in response.upper() and "CHARACTER:" not in response.upper():
        return []

    # build lookup: any name or alias (lowercase) → canonical name
    lookup: dict[str, str] = {}
    for e in roster:
        lookup[e.name.lower()] = e.name
        for a in e.aliases:
            lookup[a.lower()] = e.name

    def _resolve(raw: str) -> str | None:
        raw_l = raw.strip().lower()
        if raw_l in lookup:
            return lookup[raw_l]
        fw = _first_word(raw_l)
        for k, v in lookup.items():
            if _first_word(k) == fw:
                return v
        return None

    signals: list[_Signal] = []
    for block in re.split(r"\n---+\n?", response):
        block = block.strip()
        if not block:
            continue
        char_m   = re.search(r"^CHARACTER:\s*(.+)$",  block, re.MULTILINE)
        signal_m = re.search(r"^SIGNAL:\s*(.+)$",     block, re.MULTILINE)
        arch_m   = re.search(r"^ARCHETYPES:\s*(.+)$", block, re.MULTILINE)
        offset_m = re.search(r"^OFFSET:\s*(\d+)$",    block, re.MULTILINE)
        if not (char_m and signal_m):
            continue
        canonical = _resolve(char_m.group(1).strip())
        if canonical is None:
            continue
        scores: dict[str, float] = {}
        if arch_m:
            raw_arch = arch_m.group(1).strip()
            if raw_arch.lower() != "none":
                for part in raw_arch.split(","):
                    part = part.strip()
                    if ":" in part:
                        aid, _, score_s = part.partition(":")
                        try:
                            scores[aid.strip()] = float(score_s.strip())
                        except ValueError:
                            pass
        raw_signal = signal_m.group(1).strip().strip('"\'').strip('\u201c\u201d\u2018\u2019').strip()
        offset = max(0, min(100, int(offset_m.group(1)))) if offset_m else 50
        signals.append(_Signal(
            character=canonical,
            signal=raw_signal,
            scores=scores,
            offset=offset,
        ))
    return signals


# ── signal aggregation → character sheets ─────────────────────────────────────

def _build_sheets(
    signals: list[_Signal],
    roster:  list[RosterEntry],
    archetypes: list[dict],
) -> list[CharacterSheet]:
    archetype_name = {a["id"]: a["name"] for a in archetypes}
    all_ids        = [a["id"] for a in archetypes]

    by_char: dict[str, list[_Signal]] = {e.name: [] for e in roster}
    for sig in signals:
        if sig.character in by_char:
            by_char[sig.character].append(sig)

    sheets: list[CharacterSheet] = []
    for entry in roster:
        char_signals = by_char[entry.name]
        T = len(char_signals)
        if T == 0:
            continue

        # Sum scores over all T instances; missing scores count as 0
        totals: dict[str, float] = {aid: 0.0 for aid in all_ids}
        for sig in char_signals:
            for aid, score in sig.scores.items():
                if aid in totals:
                    totals[aid] += score

        avg = {aid: totals[aid] / T for aid in all_ids}
        ranked = sorted(avg.items(), key=lambda x: -x[1])

        top = [
            (aid, archetype_name.get(aid, aid), round(score, 2))
            for aid, score in ranked[:5]
            if score > 0
        ]

        # Evidence: for each top archetype, pick the signals where it scored highest
        evidence: dict[str, list[str]] = {}
        for aid, _, _ in top:
            scored = sorted(
                [(s.signal, s.scores.get(aid, 0.0)) for s in char_signals
                 if aid in s.scores],
                key=lambda x: -x[1],
            )
            evidence[aid] = [s for s, _ in scored[:3]]

        signals_ordered = [
            {"chunk_idx": s.chunk_idx, "offset": s.offset, "scores": dict(s.scores), "signal": s.signal}
            for s in sorted(char_signals, key=lambda s: (s.chunk_idx, s.offset))
        ]

        sheets.append(CharacterSheet(
            name=entry.name,
            aliases=entry.aliases,
            synopsis=entry.synopsis,
            top_archetypes=top,
            evidence=evidence,
            total_signals=T,
            chunks_seen=entry.chunks_seen,
            signals_ordered=signals_ordered,
        ))

    return sheets


# ── async API layer ───────────────────────────────────────────────────────────

async def _run_discovery(
    chunks:      list[str],
    client:      anthropic.AsyncAnthropic,
    model:       str,
    sem:         asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[list[RosterEntry]], TokenUsage]:
    total   = len(chunks)
    done    = [0]
    usages: list[TokenUsage] = []

    async def _call(idx: int, chunk: str) -> list[RosterEntry]:
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=1024,
                        temperature=0,
                        system=_DISCOVERY_SYSTEM,
                        messages=[{"role": "user",
                                   "content": _discovery_prompt(chunk)}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return []
            raw = resp.content[0].text.strip()
            usages.append(TokenUsage(resp.usage.input_tokens,
                                     resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Identifying characters", done[0], total)
            if on_raw:
                on_raw("discovery", idx, chunk, raw)
            return _parse_discovery(raw)

    per_chunk = list(await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks))))
    return per_chunk, sum(usages, TokenUsage())


async def _run_annotation(
    chunks:     list[str],
    roster:     list[RosterEntry],
    archetypes: list[dict],
    client:     anthropic.AsyncAnthropic,
    model:      str,
    sem:        asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system  = _annotation_system(roster, archetypes)
    total   = len(chunks)
    done    = [0]
    usages: list[TokenUsage] = []

    async def _call(idx: int, chunk: str) -> list[_Signal]:
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=2048,
                        temperature=0,
                        system=system,
                        messages=[{"role": "user",
                                   "content": _annotation_prompt(chunk)}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return []
            raw = resp.content[0].text.strip()
            usages.append(TokenUsage(resp.usage.input_tokens,
                                     resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Annotating signals", done[0], total)
            if on_raw:
                on_raw("annotation", idx, chunk, raw)
            sigs = _parse_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results     = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_signals = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_signals, sum(usages, TokenUsage())


# ── Pass 3: negative / counter-signal annotation ──────────────────────────────

def _negative_annotation_system(sheets: "list[CharacterSheet]",
                                 archetypes: list[dict]) -> str:
    arch_name = {a["id"]: a["name"] for a in archetypes}
    char_blocks = []
    for s in sheets:
        aliases = f"  (also: {', '.join(s.aliases)})" if s.aliases else ""
        arch_lines = "\n".join(
            f"      - Does NOT behave like a {arch_name.get(aid, aid)} [{aid}]."
            f" Focus on ANTI-{arch_name.get(aid, aid)} signals only —"
            f" ignore any moment consistent with {arch_name.get(aid, aid)} behaviour."
            for aid, _, _ in s.top_archetypes[:3]
        )
        char_blocks.append(f"  {s.name}{aliases}\n    watch for violations of:\n{arch_lines}")
    roster_block = "\n".join(char_blocks)

    return f"""\
You find clear, unambiguous behavioral violations of a character's dominant archetype.

Characters and their dominant archetypes (use the bracketed ID when scoring):
{roster_block}

Read the excerpt and find moments where one of these characters performs a direct, \
observable act that explicitly contradicts what their dominant archetype requires — \
a caregiver deliberately abandoning someone who needs them, a protector consciously \
betraying a person they are meant to shield, an idealist knowingly acting against \
their stated principles for personal gain.

STRICT EXCLUSIONS — do not flag any of these:
- Ambivalence, doubt, tiredness, or momentary fear
- Internal conflict or mixed feelings without a concrete contradicting act
- Vulnerability, self-sacrifice, or moments of quiet complexity
- Any moment that could be read as consistent with the archetype under a charitable reading

Only flag a moment if a neutral reader presented with the quote alone would \
immediately recognise it as an unambiguous violation of the listed archetype.

For each clear violation:
1. Describe the specific act and why it is an unambiguous contradiction.
2. Score how severely the act violates each dominant archetype listed above \
(0 = no violation, 10 = complete inversion of the archetype). \
Only list archetypes scoring 4 or higher; if none reach 4, output ARCHETYPES: none.

Output one block per violation:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence verbatim quote or tight paraphrase of the violating act]
BEHAVIOR: [one sentence naming the specific act and the archetype it violates]
ARCHETYPES: [id:score, id:score, ...] or "none"
OFFSET: [0-100, where 0 = start of excerpt, 100 = end]
---

If no clear unambiguous violations appear in this excerpt, output: NO SIGNALS
No text outside the structured blocks."""


async def _run_negative_annotation(
    chunks:     list[str],
    sheets:     "list[CharacterSheet]",
    archetypes: list[dict],
    client:     anthropic.AsyncAnthropic,
    model:      str,
    sem:        asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system  = _negative_annotation_system(sheets, archetypes)
    roster  = [RosterEntry(name=s.name, aliases=s.aliases, synopsis=s.synopsis)
               for s in sheets]
    total   = len(chunks)
    done    = [0]
    usages: list[TokenUsage] = []

    async def _call(idx: int, chunk: str) -> list[_Signal]:
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=2048,
                        temperature=0,
                        system=system,
                        messages=[{"role": "user",
                                   "content": _annotation_prompt(chunk)}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return []
            raw = resp.content[0].text.strip()
            usages.append(TokenUsage(resp.usage.input_tokens,
                                     resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Scanning contradictions", done[0], total)
            if on_raw:
                on_raw("negative", idx, chunk, raw)
            sigs = _parse_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results  = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_sigs = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_sigs, sum(usages, TokenUsage())


def _merge_neg_signals(sheets: "list[CharacterSheet]",
                        neg_signals: list[_Signal]) -> None:
    by_char: dict[str, list[_Signal]] = {s.name: [] for s in sheets}
    for sig in neg_signals:
        if sig.character in by_char:
            by_char[sig.character].append(sig)
    for sheet in sheets:
        sheet.neg_signals_ordered = [
            {"chunk_idx": s.chunk_idx, "offset": s.offset,
             "scores": {aid: sc for aid, sc in s.scores.items() if sc >= 4},
             "signal": s.signal}
            for s in sorted(by_char[sheet.name], key=lambda s: (s.chunk_idx, s.offset))
            if any(sc >= 4 for sc in s.scores.values())
        ]


async def _analyze_async(
    text:               str,
    model:              str,
    roster_size:        int,
    on_progress,
    synopsis:           str,
    focal_characters:   list[str] | None = None,
    enabled_archetypes: list[str] | None = None,
    extra_archetypes:   list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    all_archetypes = load_archetypes()
    if extra_archetypes:
        all_archetypes = all_archetypes + extra_archetypes
    archetypes = (
        [a for a in all_archetypes if a["id"] in enabled_archetypes]
        if enabled_archetypes is not None
        else all_archetypes
    )
    if not archetypes:
        archetypes = all_archetypes   # safety: never send an empty list

    client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    # Pre-compute chunk counts so we can report unified progress across all passes
    discovery_chunks  = _chunk(text, DISCOVERY_CHUNK_CHARS) if not focal_characters else []
    annotation_chunks = _chunk(text, ANNOTATION_CHUNK_CHARS)
    disc_n  = len(discovery_chunks)
    annot_n = len(annotation_chunks)
    total_steps = (disc_n + 1 if not focal_characters else 0) + annot_n * 2
    done = [0]

    def _prog(*_args):
        done[0] += 1
        if on_progress:
            on_progress(done[0], total_steps)

    if focal_characters:
        roster  = [RosterEntry(name=n, aliases=[], synopsis="", chunks_seen=1)
                   for n in focal_characters]
        usage1  = TokenUsage()
    else:
        per_chunk, usage1 = await _run_discovery(
            discovery_chunks, client, model, sem, _prog, on_raw
        )
        _prog()  # reconciliation call
        roster = await _aggregate_roster(per_chunk, k=roster_size, client=client, model=model, synopsis=synopsis)
        if not roster:
            return [], usage1

    all_signals, usage2 = await _run_annotation(
        annotation_chunks, roster, archetypes, client, model, sem, _prog, on_raw
    )

    sheets = _build_sheets(all_signals, roster, archetypes)

    if sheets:
        neg_signals, usage3 = await _run_negative_annotation(
            annotation_chunks, sheets, archetypes, client, model, sem, _prog, on_raw
        )
        _merge_neg_signals(sheets, neg_signals)
    else:
        usage3 = TokenUsage()

    return sheets, usage1 + usage2 + usage3


# ── public API ────────────────────────────────────────────────────────────────

def analyze_characters(
    filepath:           str,
    synopsis:           str,
    model:              str = DEFAULT_MODEL,
    roster_size:        int = DEFAULT_ROSTER_SIZE,
    on_progress=None,
    focal_characters:   list[str] | None = None,
    enabled_archetypes: list[str] | None = None,
    extra_archetypes:   list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    """
    Run character analysis on a .docx file.

    focal_characters   — skip Pass 1 and use these names as the roster.
    enabled_archetypes — restrict scoring to these archetype IDs (None = all).
    extra_archetypes   — additional archetype dicts (id, name, description) to include.
    on_progress(phase, current, total) — called after each API call.
    on_raw(phase, chunk_idx, chunk_text, raw_response) — called with every raw
        LLM response; useful for logging and prompt evaluation.
    """
    text = load_docx(filepath)
    if not text.strip():
        return [], TokenUsage()
    return asyncio.run(
        _analyze_async(
            text, model, roster_size, on_progress, synopsis,
            focal_characters, enabled_archetypes, extra_archetypes, on_raw,
        )
    )
