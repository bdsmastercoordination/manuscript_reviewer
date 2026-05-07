"""
Three-pass Big Five personality analysis for Mirror.

Identical pipeline structure to emotion_analyzer but scores character
personality traits (Openness, Conscientiousness, Extraversion, Agreeableness,
Neuroticism) instead of emotions.

Pass 1  Character discovery -- shared with character_analyzer.
Pass 2  Trait annotation -- finds and scores Big Five trait expressions.
Pass 3  Contradiction annotation -- finds moments of acting against dominant traits.
"""

import json
import asyncio
import re
from pathlib import Path

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx
from .character_analyzer import (
    RosterEntry, CharacterSheet, _Signal, _chunk, _first_word,
    _aggregate_roster, _run_discovery,
    DISCOVERY_CHUNK_CHARS, ANNOTATION_CHUNK_CHARS,
    DEFAULT_ROSTER_SIZE, MAX_CONCURRENT, DEFAULT_MODEL,
)

PERSONALITY_PATH = Path(__file__).parent.parent / "config" / "personality.jsonl"


def load_personality_traits() -> list[dict]:
    with open(PERSONALITY_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Pass 2: trait annotation ──────────────────────────────────────────────────

def _trait_annotation_system(roster: list[RosterEntry], traits: list[dict]) -> str:
    roster_block = "\n".join(
        f"  {e.name}" + (f"  (also: {', '.join(e.aliases)})" if e.aliases else "")
        for e in roster
    )
    trait_block = "\n".join(
        f"  {t['id']}: {t.get('high_name', t['name'])} (+3) ↔ {t.get('low_name', 'Low')} (-3)\n"
        f"    +: {t['description']}\n"
        f"    -: {t.get('low_description', 'Opposite tendency.')}"
        for t in traits
    )
    return f"""\
You annotate fiction manuscripts for Big Five personality trait expression using a bipolar scale.

Characters to watch for:
{roster_block}

Read the excerpt and find every moment where one of these characters clearly \
expresses a personality trait -- including subtle, understated, or ambiguous moments.

Each Big Five trait is a bipolar dimension. Score each on a scale from -3 to +3:
  +3 = strongly expresses the HIGH pole (e.g. very extraverted)
  -3 = strongly expresses the LOW pole (e.g. very introverted)
   0 = no evidence either way

Traits (high pole ↔ low pole):
{trait_block}

For each moment:
1. Describe what the character does and how it reveals their personality.
2. Score each relevant trait from -3 to +3. List only traits with |score| >= 1; \
if none qualify, output TRAITS: none.

Output one block per moment:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence quote or tight paraphrase of the moment]
BEHAVIOR: [your one-sentence reflection on which pole this moment expresses and why]
TRAITS: [id:score, id:score, ...] or "none"
OFFSET: [0-100, where 0 = start of excerpt, 100 = end]
---

If no listed characters appear in this excerpt at all, output: NO SIGNALS
No text outside the structured blocks."""


def _parse_trait_signals(response: str, roster: list[RosterEntry]) -> list[_Signal]:
    if "NO SIGNALS" in response.upper() and "CHARACTER:" not in response.upper():
        return []

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
        char_m   = re.search(r"^CHARACTER:\s*(.+)$", block, re.MULTILINE)
        signal_m = re.search(r"^SIGNAL:\s*(.+)$",    block, re.MULTILINE)
        trait_m  = re.search(r"^TRAITS:\s*(.+)$",    block, re.MULTILINE)
        offset_m = re.search(r"^OFFSET:\s*(\d+)$",   block, re.MULTILINE)
        if not (char_m and signal_m):
            continue
        canonical = _resolve(char_m.group(1).strip())
        if canonical is None:
            continue
        scores: dict[str, float] = {}
        if trait_m:
            raw_traits = trait_m.group(1).strip()
            if raw_traits.lower() != "none":
                for part in raw_traits.split(","):
                    part = part.strip()
                    if ":" in part:
                        tid, _, score_s = part.partition(":")
                        try:
                            scores[tid.strip()] = float(score_s.strip())
                        except ValueError:
                            pass
        raw_signal = signal_m.group(1).strip().strip("\"'").strip('“”‘’').strip()
        offset = max(0, min(100, int(offset_m.group(1)))) if offset_m else 50
        signals.append(_Signal(character=canonical, signal=raw_signal, scores=scores, offset=offset))
    return signals


def _build_personality_sheets(
    signals: list[_Signal],
    roster: list[RosterEntry],
    traits: list[dict],
) -> list[CharacterSheet]:
    trait_name = {t["id"]: t["name"] for t in traits}
    all_ids = [t["id"] for t in traits]

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

        totals: dict[str, float] = {tid: 0.0 for tid in all_ids}
        for sig in char_signals:
            for tid, score in sig.scores.items():
                if tid in totals:
                    totals[tid] += score

        avg = {tid: totals[tid] / T for tid in all_ids}
        # rank by |avg| so strongest poles (either direction) come first
        ranked = sorted(avg.items(), key=lambda x: -abs(x[1]))

        top = [
            (tid, trait_name.get(tid, tid), round(score, 2))
            for tid, score in ranked[:5]
            if abs(score) >= 0.1
        ]

        evidence: dict[str, list[str]] = {}
        for tid, _, _ in top:
            scored = sorted(
                [(s.signal, s.scores.get(tid, 0.0)) for s in char_signals
                 if tid in s.scores],
                key=lambda x: -abs(x[1]),
            )
            evidence[tid] = [s for s, _ in scored[:3]]

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


async def _run_trait_annotation(
    chunks: list[str],
    roster: list[RosterEntry],
    traits: list[dict],
    client: anthropic.AsyncAnthropic,
    model: str,
    sem: asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system = _trait_annotation_system(roster, traits)
    total  = len(chunks)
    done   = [0]
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
                                   "content": f"Annotate this excerpt:\n\n{chunk}"}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return []
            raw = resp.content[0].text.strip()
            usages.append(TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Annotating personality", done[0], total)
            if on_raw:
                on_raw("personality", idx, chunk, raw)
            sigs = _parse_trait_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_signals = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_signals, sum(usages, TokenUsage())


# ── Pass 3: contradiction annotation ─────────────────────────────────────────

def _contradiction_annotation_system(sheets: list[CharacterSheet],
                                      traits: list[dict]) -> str:
    trait_name  = {t["id"]: t["name"]      for t in traits}
    high_name   = {t["id"]: t.get("high_name", t["name"])       for t in traits}
    low_name    = {t["id"]: t.get("low_name",  "Low " + t["name"]) for t in traits}
    char_blocks = []
    for s in sheets:
        aliases = f"  (also: {', '.join(s.aliases)})" if s.aliases else ""
        top_parts = []
        for tid, _, score in s.top_archetypes[:3]:
            pole = high_name.get(tid, tid) if score >= 0 else low_name.get(tid, tid)
            top_parts.append(f"{pole} [{tid}]")
        top = "  ".join(top_parts)
        char_blocks.append(f"  {s.name}{aliases}\n    dominant poles: {top}")
    roster_block = "\n".join(char_blocks)

    return f"""\
You find moments where characters act against or contradict their dominant personality poles.

Characters and their dominant poles (use the bracketed trait ID when scoring):
{roster_block}

Read the excerpt and find every moment where one of these characters behaves in a way \
that contradicts their dominant pole — a conscientious character acting recklessly, \
an agreeable character being callous, an introvert seeking out a crowd, \
an open character refusing a new idea.

Score each contradiction on the same -3 to +3 bipolar scale used for annotations: \
assign the score that reflects the CONTRADICTING behavior (the opposite of their dominant pole). \
For example, if a conscientious character acts carelessly, score conscientiousness: -2. \
List only traits with |score| >= 1; if none qualify, output TRAITS: none.

Output one block per moment:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence quote or tight paraphrase of the moment]
BEHAVIOR: [one sentence on how this contradicts their dominant personality pole]
TRAITS: [id:score, id:score, ...] or "none"
---

If no contradiction moments appear in this excerpt, output: NO SIGNALS
No text outside the structured blocks."""


async def _run_contradiction_annotation(
    chunks: list[str],
    sheets: list[CharacterSheet],
    traits: list[dict],
    client: anthropic.AsyncAnthropic,
    model: str,
    sem: asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system = _contradiction_annotation_system(sheets, traits)
    roster = [RosterEntry(name=s.name, aliases=s.aliases, synopsis=s.synopsis)
              for s in sheets]
    total  = len(chunks)
    done   = [0]
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
                                   "content": f"Annotate this excerpt:\n\n{chunk}"}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return []
            raw = resp.content[0].text.strip()
            usages.append(TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Scanning trait contradictions", done[0], total)
            if on_raw:
                on_raw("contradiction", idx, chunk, raw)
            sigs = _parse_trait_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_sigs = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_sigs, sum(usages, TokenUsage())


def _merge_contradiction_signals(sheets: list[CharacterSheet],
                                  neg_signals: list[_Signal]) -> None:
    by_char: dict[str, list[_Signal]] = {s.name: [] for s in sheets}
    for sig in neg_signals:
        if sig.character in by_char:
            by_char[sig.character].append(sig)
    for sheet in sheets:
        sheet.neg_signals_ordered = [
            {"chunk_idx": s.chunk_idx, "scores": dict(s.scores), "signal": s.signal}
            for s in sorted(by_char[sheet.name], key=lambda s: s.chunk_idx)
        ]


# ── async orchestration ───────────────────────────────────────────────────────

async def _analyze_async(
    text: str,
    model: str,
    roster_size: int,
    on_progress,
    synopsis: str,
    focal_characters: list[str] | None = None,
    enabled_traits: list[str] | None = None,
    extra_traits: list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    all_traits = load_personality_traits()
    if extra_traits:
        all_traits = all_traits + extra_traits
    traits = (
        [t for t in all_traits if t["id"] in enabled_traits]
        if enabled_traits is not None
        else all_traits
    )
    if not traits:
        traits = all_traits

    client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    discovery_chunks  = _chunk(text, DISCOVERY_CHUNK_CHARS) if not focal_characters else []
    annotation_chunks = _chunk(text, ANNOTATION_CHUNK_CHARS)
    disc_n  = len(discovery_chunks)
    annot_n = len(annotation_chunks)
    total_steps = (disc_n + 1 if not focal_characters else 0) + annot_n
    done = [0]

    def _prog(*_args):
        done[0] += 1
        if on_progress:
            on_progress(done[0], total_steps)

    if focal_characters:
        roster = [RosterEntry(name=n, aliases=[], synopsis="", chunks_seen=1)
                  for n in focal_characters]
        usage1 = TokenUsage()
    else:
        per_chunk, usage1 = await _run_discovery(
            discovery_chunks, client, model, sem, _prog, on_raw
        )
        _prog()
        roster = await _aggregate_roster(per_chunk, k=roster_size, client=client, model=model, synopsis=synopsis)
        if not roster:
            return [], usage1

    all_signals, usage2 = await _run_trait_annotation(
        annotation_chunks, roster, traits, client, model, sem, _prog, on_raw
    )

    sheets = _build_personality_sheets(all_signals, roster, traits)

    return sheets, usage1 + usage2


# ── public API ────────────────────────────────────────────────────────────────

def analyze_personality(
    filepath: str,
    synopsis: str,
    model: str = DEFAULT_MODEL,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    on_progress=None,
    focal_characters: list[str] | None = None,
    enabled_traits: list[str] | None = None,
    extra_traits: list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    """
    Run Big Five personality analysis on a .docx file.

    focal_characters -- skip Pass 1 and use these names as the roster.
    enabled_traits   -- restrict scoring to these trait IDs (None = all).
    extra_traits     -- additional trait dicts (id, name, description) to include.
    on_progress(current, total) -- called after each API call.
    """
    text = load_docx(filepath)
    if not text.strip():
        return [], TokenUsage()
    return asyncio.run(
        _analyze_async(
            text, model, roster_size, on_progress, synopsis,
            focal_characters, enabled_traits, extra_traits, on_raw,
        )
    )
