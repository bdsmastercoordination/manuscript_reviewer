"""
Three-pass emotion analysis for Mirror.

Identical pipeline structure to character_analyzer but scores character
emotions (Joy, Sadness, Anger, Fear, Love, Shame, Disgust, Surprise,
Longing, Pride) instead of narrative archetypes.

Pass 1  Character discovery -- shared with character_analyzer.
Pass 2  Emotion annotation -- finds and scores emotional moments.
Pass 3  Suppression annotation -- finds moments of emotional masking.
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

EMOTIONS_PATH = Path(__file__).parent.parent / "config" / "emotions.jsonl"


def load_emotions() -> list[dict]:
    with open(EMOTIONS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Pass 2: emotion annotation ────────────────────────────────────────────────

def _emotion_annotation_system(roster: list[RosterEntry], emotions: list[dict]) -> str:
    roster_block = "\n".join(
        f"  {e.name}" + (f"  (also: {', '.join(e.aliases)})" if e.aliases else "")
        for e in roster
    )
    emotion_block = "\n".join(
        f"  {e['id']}: {e['name']} -- {e['description']}"
        for e in emotions
    )
    return f"""\
You annotate fiction manuscripts for character emotion.

Characters to watch for:
{roster_block}

Read the excerpt and find every moment where one of these characters feels, \
expresses, or visibly suppresses an emotion -- including subtle, quiet, or \
ambiguous moments, not only dramatic ones. A character holding back tears, \
forcing a smile, or going very still counts as much as open weeping or rage.

For each moment, work in two steps:
1. Describe what the character feels or how they express (or hide) their \
emotion in your own words.
2. Score how strongly each emotion below fits that specific moment \
(0 = not present, 10 = overwhelmingly dominant). \
List only emotions scoring 3 or higher; if none reach 3, output EMOTIONS: none.

Emotions:
{emotion_block}

Output one block per moment:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence quote or tight paraphrase of the moment]
BEHAVIOR: [your one-sentence reflection on what this emotional moment reveals about the character]
EMOTIONS: [id:score, id:score, ...] or "none"
OFFSET: [0-100, where 0 = start of excerpt, 100 = end]
---

If no listed characters appear in this excerpt at all, output: NO SIGNALS
No text outside the structured blocks."""


def _parse_emotion_signals(response: str, roster: list[RosterEntry]) -> list[_Signal]:
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
        emot_m   = re.search(r"^EMOTIONS:\s*(.+)$",  block, re.MULTILINE)
        offset_m = re.search(r"^OFFSET:\s*(\d+)$",   block, re.MULTILINE)
        if not (char_m and signal_m):
            continue
        canonical = _resolve(char_m.group(1).strip())
        if canonical is None:
            continue
        scores: dict[str, float] = {}
        if emot_m:
            raw_emot = emot_m.group(1).strip()
            if raw_emot.lower() != "none":
                for part in raw_emot.split(","):
                    part = part.strip()
                    if ":" in part:
                        eid, _, score_s = part.partition(":")
                        try:
                            scores[eid.strip()] = float(score_s.strip())
                        except ValueError:
                            pass
        raw_signal = signal_m.group(1).strip().strip("\"'").strip('\u201c\u201d\u2018\u2019').strip()
        offset = max(0, min(100, int(offset_m.group(1)))) if offset_m else 50
        signals.append(_Signal(character=canonical, signal=raw_signal, scores=scores, offset=offset))
    return signals


def _build_emotion_sheets(
    signals: list[_Signal],
    roster: list[RosterEntry],
    emotions: list[dict],
) -> list[CharacterSheet]:
    emotion_name = {e["id"]: e["name"] for e in emotions}
    all_ids = [e["id"] for e in emotions]

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

        totals: dict[str, float] = {eid: 0.0 for eid in all_ids}
        for sig in char_signals:
            for eid, score in sig.scores.items():
                if eid in totals:
                    totals[eid] += score

        avg = {eid: totals[eid] / T for eid in all_ids}
        ranked = sorted(avg.items(), key=lambda x: -x[1])

        top = [
            (eid, emotion_name.get(eid, eid), round(score, 2))
            for eid, score in ranked[:5]
            if score > 0
        ]

        evidence: dict[str, list[str]] = {}
        for eid, _, _ in top:
            scored = sorted(
                [(s.signal, s.scores.get(eid, 0.0)) for s in char_signals
                 if eid in s.scores],
                key=lambda x: -x[1],
            )
            evidence[eid] = [s for s, _ in scored[:3]]

        signals_ordered = [
            {"chunk_idx": s.chunk_idx, "offset": s.offset, "scores": dict(s.scores), "signal": s.signal}
            for s in sorted(char_signals, key=lambda s: (s.chunk_idx, s.offset))
        ]

        sheets.append(CharacterSheet(
            name=entry.name,
            aliases=entry.aliases,
            synopsis=entry.synopsis,
            top_archetypes=top,   # reusing field; stores emotion data
            evidence=evidence,
            total_signals=T,
            chunks_seen=entry.chunks_seen,
            signals_ordered=signals_ordered,
        ))
    return sheets


async def _run_emotion_annotation(
    chunks: list[str],
    roster: list[RosterEntry],
    emotions: list[dict],
    client: anthropic.AsyncAnthropic,
    model: str,
    sem: asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system = _emotion_annotation_system(roster, emotions)
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
                on_progress("Annotating emotions", done[0], total)
            if on_raw:
                on_raw("annotation", idx, chunk, raw)
            sigs = _parse_emotion_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_signals = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_signals, sum(usages, TokenUsage())


# ── Pass 3: emotion suppression annotation ────────────────────────────────────

def _suppression_annotation_system(sheets: list[CharacterSheet],
                                    emotions: list[dict]) -> str:
    emotion_name = {e["id"]: e["name"] for e in emotions}
    char_blocks = []
    for s in sheets:
        aliases = f"  (also: {', '.join(s.aliases)})" if s.aliases else ""
        top = "  ".join(
            f"{emotion_name.get(eid, eid)} [{eid}]"
            for eid, _, _ in s.top_archetypes[:3]
        )
        char_blocks.append(f"  {s.name}{aliases}\n    dominant emotions: {top}")
    roster_block = "\n".join(char_blocks)

    return f"""\
You find moments where characters suppress, mask, or contradict their dominant emotions.

Characters and their dominant emotions (use the bracketed ID when scoring):
{roster_block}

Read the excerpt and find every moment where one of these characters suppresses, \
hides, or acts against their dominant emotions -- putting on a brave face despite grief, \
feigning indifference when afraid, masking love with coldness, channelling sadness into anger, \
laughing to avoid crying.

For each such moment:
1. Describe what the character does and how it masks or contradicts their dominant emotion.
2. Score how strongly the moment suppresses each dominant emotion listed above \
(0 = no suppression, 10 = completely masks or inverts this emotion). \
List only emotions scoring 3 or higher; if none reach 3, output EMOTIONS: none.

Output one block per moment:

CHARACTER: [canonical name from the list above]
SIGNAL: [one-sentence quote or tight paraphrase of the moment]
BEHAVIOR: [one sentence on how this suppresses or contradicts their dominant emotion]
EMOTIONS: [id:score, id:score, ...] or "none"
---

If no suppression moments appear in this excerpt, output: NO SIGNALS
No text outside the structured blocks."""


async def _run_suppression_annotation(
    chunks: list[str],
    sheets: list[CharacterSheet],
    emotions: list[dict],
    client: anthropic.AsyncAnthropic,
    model: str,
    sem: asyncio.Semaphore,
    on_progress,
    on_raw=None,
) -> tuple[list[_Signal], TokenUsage]:
    system = _suppression_annotation_system(sheets, emotions)
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
                on_progress("Scanning emotion suppression", done[0], total)
            if on_raw:
                on_raw("suppression", idx, chunk, raw)
            sigs = _parse_emotion_signals(raw, roster)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_sigs = [sig for chunk_sigs in results for sig in chunk_sigs]
    return all_sigs, sum(usages, TokenUsage())


def _merge_suppression_signals(sheets: list[CharacterSheet],
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
    enabled_emotions: list[str] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    all_emotions = load_emotions()
    emotions = (
        [e for e in all_emotions if e["id"] in enabled_emotions]
        if enabled_emotions is not None
        else all_emotions
    )
    if not emotions:
        emotions = all_emotions

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
        _prog()  # reconciliation call
        roster = await _aggregate_roster(per_chunk, k=roster_size, client=client, model=model, synopsis=synopsis)
        if not roster:
            return [], usage1

    all_signals, usage2 = await _run_emotion_annotation(
        annotation_chunks, roster, emotions, client, model, sem, _prog, on_raw
    )

    sheets = _build_emotion_sheets(all_signals, roster, emotions)

    return sheets, usage1 + usage2


# ── public API ────────────────────────────────────────────────────────────────

def analyze_emotions(
    filepath: str,
    synopsis: str,
    model: str = DEFAULT_MODEL,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    on_progress=None,
    focal_characters: list[str] | None = None,
    enabled_emotions: list[str] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    """
    Run emotion analysis on a .docx file.

    focal_characters  -- skip Pass 1 and use these names as the roster.
    enabled_emotions  -- restrict scoring to these emotion IDs (None = all).
    on_progress(phase, current, total) -- called after each API call.
    """
    text = load_docx(filepath)
    if not text.strip():
        return [], TokenUsage()
    return asyncio.run(
        _analyze_async(
            text, model, roster_size, on_progress, synopsis,
            focal_characters, enabled_emotions, on_raw,
        )
    )
