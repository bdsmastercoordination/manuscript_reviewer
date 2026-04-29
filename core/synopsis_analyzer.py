"""
Two-pass synopsis + beat-label analysis for Mirror.

Pass 1  Parallel beat notes per large chunk.
Pass 2  Single synthesis call — logline + synopsis paragraph.
Pass 3  Single call over beat notes — story-craft labels with summaries.
"""

import asyncio
import math
import re
from dataclasses import dataclass, field

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx
from .character_analyzer import _chunk, DISCOVERY_CHUNK_CHARS, MAX_CONCURRENT, DEFAULT_MODEL


@dataclass
class PlotBeat:
    term:     str
    beat_idx: int
    summary:  str
    detail:   str


@dataclass
class SynopsisResult:
    logline:    str
    synopsis:   str
    beat_notes: list[str]      = field(default_factory=list)
    plot_beats: list[PlotBeat] = field(default_factory=list)


# ── Pass 1: beat notes ────────────────────────────────────────────────────────

_BEAT_SYSTEM = """\
You are an editorial analyst summarizing a fiction manuscript chunk by chunk.

Read the excerpt and produce 3-5 bullet points capturing:
* Which named characters appear and what they do or say
* The key event, decision, or turning point (if any)
* The emotional register of the scene -- tension, tenderness, dread, levity, etc.

Rules:
- Be specific and concrete. Quote character names directly.
- One tight line per bullet. No sub-bullets.
- Do not speculate beyond what is on the page.
- If the excerpt is transitional or purely descriptive with no clear event, note that briefly.

Output only the bullets. No header, no commentary."""


async def _run_beat_notes(
    chunks: list[str],
    client: anthropic.AsyncAnthropic,
    model: str,
    sem: asyncio.Semaphore,
    on_progress,
) -> tuple[list[str], TokenUsage]:
    total  = len(chunks)
    done   = [0]
    usages: list[TokenUsage] = []

    async def _call(chunk: str) -> str:
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=512,
                        temperature=0,
                        system=_BEAT_SYSTEM,
                        messages=[{"role": "user",
                                   "content": f"Summarize this excerpt:\n\n{chunk}"}],
                    )
                    break
                except anthropic.RateLimitError:
                    await asyncio.sleep(20 * (2 ** attempt))
            else:
                return ""
            usages.append(TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens))
            done[0] += 1
            if on_progress:
                on_progress("Reading manuscript", done[0], total)
            return resp.content[0].text.strip()

    notes = await asyncio.gather(*(_call(c) for c in chunks))
    return list(notes), sum(usages, TokenUsage())


# ── Pass 2: synthesis ─────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are an editorial analyst. Below are beat notes from a fiction manuscript, \
in narrative order. Synthesize them into two sections:

LOGLINE: One sentence (under 25 words) capturing the core dramatic premise \
and who it concerns. No spoilers of the ending.
SYNOPSIS: One paragraph (4-6 sentences) summarizing the full narrative arc, \
including how it resolves.

Output each label in ALL CAPS followed by a colon, then the content. \
No other text."""


def _parse_synthesis(text: str) -> SynopsisResult:
    def _extract(label: str) -> str:
        m = re.search(rf"^{label}:\s*(.+?)(?=\n[A-Z]+:|$)", text,
                      re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    return SynopsisResult(
        logline=_extract("LOGLINE"),
        synopsis=_extract("SYNOPSIS"),
    )


async def _synthesize(
    beat_notes: list[str],
    client: anthropic.AsyncAnthropic,
    model: str,
    on_progress,
) -> tuple[SynopsisResult, TokenUsage]:
    numbered = "\n\n".join(
        f"[Beat {i+1}]\n{note}" for i, note in enumerate(beat_notes) if note.strip()
    )
    for attempt in range(5):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user",
                           "content": f"Beat notes:\n\n{numbered}"}],
            )
            break
        except anthropic.RateLimitError:
            await asyncio.sleep(20 * (2 ** attempt))
    else:
        return SynopsisResult(logline="", synopsis=""), TokenUsage()

    if on_progress:
        on_progress("Synthesizing", 1, 1)
    usage  = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    result = _parse_synthesis(resp.content[0].text.strip())
    return result, usage


# ── Pass 3: beat labelling ────────────────────────────────────────────────────

_BEAT_LABEL_SYSTEM = """\
You are an editorial analyst. Below are numbered beat notes from a fiction manuscript.

For each beat that represents a clear narrative function, assign a common story-craft \
term and write a brief description. Examples of terms (not exhaustive): inciting incident, \
meet-cute, status quo, call to adventure, midpoint, dark night of the soul, climax, \
false victory, new character introduction, revelation, confrontation, resolution, twist, \
turning point, romantic tension, comic relief, black moment, reunion. Use your own label \
if none of the examples fit. Skip purely transitional or descriptive beats with no \
clear story function.

Output one block per labelled beat:

TERM: [story-craft label]
BEAT: [beat number]
SUMMARY: [one sentence, under 20 words]
DETAIL: [2-4 sentences expanding on the beat]
---

No text outside the structured blocks."""


def _parse_beat_labels(text: str) -> list[PlotBeat]:
    beats: list[PlotBeat] = []
    for block in re.split(r"\n---+\n?", text):
        block = block.strip()
        if not block:
            continue
        term_m    = re.search(r"^TERM:\s*(.+)$",    block, re.MULTILINE)
        beat_m    = re.search(r"^BEAT:\s*(\d+)$",   block, re.MULTILINE)
        summary_m = re.search(r"^SUMMARY:\s*(.+)$", block, re.MULTILINE)
        detail_m  = re.search(r"^DETAIL:\s*(.+?)(?=\n[A-Z]+:|$)", block,
                               re.MULTILINE | re.DOTALL)
        if not (term_m and summary_m):
            continue
        beats.append(PlotBeat(
            term=term_m.group(1).strip(),
            beat_idx=int(beat_m.group(1)) - 1 if beat_m else 0,
            summary=summary_m.group(1).strip(),
            detail=detail_m.group(1).strip() if detail_m else "",
        ))
    return beats


LABEL_SPLIT_THRESHOLD = 40


async def _label_beats_call(
    indexed_notes: list[tuple[int, str]],
    client: anthropic.AsyncAnthropic,
    model: str,
) -> tuple[list[PlotBeat], TokenUsage]:
    numbered = "\n\n".join(
        f"[Beat {orig_idx+1}]\n{note}"
        for orig_idx, note in indexed_notes
        if note.strip()
    )
    for attempt in range(5):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0,
                system=_BEAT_LABEL_SYSTEM,
                messages=[{"role": "user",
                           "content": f"Beat notes:\n\n{numbered}"}],
            )
            break
        except anthropic.RateLimitError:
            await asyncio.sleep(20 * (2 ** attempt))
    else:
        return [], TokenUsage()
    usage = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    beats = _parse_beat_labels(resp.content[0].text.strip())
    return beats, usage


async def _label_beats(
    beat_notes: list[str],
    client: anthropic.AsyncAnthropic,
    model: str,
    on_progress,
) -> tuple[list[PlotBeat], TokenUsage]:
    n       = len(beat_notes)
    n_calls = max(1, math.ceil(n / LABEL_SPLIT_THRESHOLD))
    size    = math.ceil(n / n_calls)

    all_beats: list[PlotBeat] = []
    total_usage = TokenUsage()

    for call_idx in range(n_calls):
        start = call_idx * size
        end   = min(start + size, n)
        slice_notes = [(i, beat_notes[i]) for i in range(start, end)]
        beats, usage = await _label_beats_call(slice_notes, client, model)
        all_beats.extend(beats)
        total_usage = total_usage + usage
        if on_progress:
            on_progress("Labelling beats", call_idx + 1, n_calls)

    all_beats.sort(key=lambda b: b.beat_idx)
    return all_beats, total_usage


# ── orchestration ─────────────────────────────────────────────────────────────

async def _analyze_async(
    text: str,
    model: str,
    on_progress,
) -> tuple[SynopsisResult, TokenUsage]:
    client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    sem    = asyncio.Semaphore(MAX_CONCURRENT)
    chunks = _chunk(text, DISCOVERY_CHUNK_CHARS)

    beat_notes, usage1 = await _run_beat_notes(chunks, client, model, sem, on_progress)
    result, usage2     = await _synthesize(beat_notes, client, model, on_progress)
    result.beat_notes  = beat_notes
    plot_beats, usage3 = await _label_beats(beat_notes, client, model, on_progress)
    result.plot_beats  = plot_beats
    return result, usage1 + usage2 + usage3


# ── public API ────────────────────────────────────────────────────────────────

def analyze_synopsis(
    filepath: str,
    model: str = DEFAULT_MODEL,
    on_progress=None,
) -> tuple[SynopsisResult, TokenUsage]:
    """Run synopsis analysis on a .docx file. Returns (result, usage)."""
    text = load_docx(filepath)
    if not text.strip():
        return SynopsisResult(logline="", synopsis=""), TokenUsage()
    return asyncio.run(_analyze_async(text, model, on_progress))
