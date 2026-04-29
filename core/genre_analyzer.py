"""
Genre detection for Mirror.

Single-pass annotation that identifies genre tropes in each text chunk.
Returns GenreResult objects sorted by total signal count.
"""

import json
import asyncio
import re
from pathlib import Path
from dataclasses import dataclass, field

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx
from .character_analyzer import (
    _chunk, MAX_CONCURRENT, DEFAULT_MODEL, ANNOTATION_CHUNK_CHARS,
)

GENRES_PATH = Path(__file__).parent.parent / "config" / "genres.json"

def load_genres() -> list[dict]:
    with open(GENRES_PATH, encoding="utf-8") as f:
        return json.load(f)["genres"]


@dataclass
class TropeSignal:
    genre: str
    trope: str
    signal: str
    chunk_idx: int = 0


@dataclass
class GenreResult:
    genre: str
    total_signals: int
    trope_signals: list = field(default_factory=list)


# ── prompting ─────────────────────────────────────────────────────────────────

def _annotation_system(genres: list[dict]) -> str:
    genre_block = "\n".join(
        f"  {g['name']}: {', '.join(g['tropes'])}"
        for g in genres
    )
    return f"""\
You annotate fiction manuscripts for genre tropes.

Read the excerpt and identify every moment, pattern, or element that signals \
a specific genre trope. Include subtle signals, not just obvious ones. \
A single excerpt may yield signals for multiple genres.

CRITICAL RULES — violating any of these invalidates the signal:
1. Every signal must be grounded in the supplied excerpt only. Do not draw on \
knowledge of the full source text. If you cannot quote the evidence directly \
from this passage, suppress the signal entirely.
2. The SIGNAL field must contain a verbatim quotation from the excerpt as \
evidence — paraphrase alone is not sufficient.
3. A trope must describe a stable, defining narrative or character pattern, \
not a single situational reaction (shock, silence, one-off deflection).
4. "unreliable narrator" requires demonstrable distortion or self-contradiction \
by the narrator. Ironic editorializing, direct reader address, and mock-modesty \
are an "ironic narrator" / authorial intrusion — do NOT label them unreliable.
5. Magical Realism requires literal magic treated as mundane reality. Hyperbolic \
language, symbolic objects, and rhetorical flourish are NOT magical realism.
6. Romance tropes (enemies to lovers, grumpy/sunshine, love triangle) require \
an established antagonism or contrast between two named characters. Do not \
project these onto an already-loving couple or a character described alone.
7. "family dysfunction" requires relational breakdown or pathology within the \
family unit — poverty or economic hardship alone does not qualify.
8. "coming of age" requires a protagonist transitioning from youth/inexperience \
to maturity as a central arc — do not apply it to established adults making \
a single sacrifice or moral choice.

For each signal decide:
1. Which genre it belongs to (from the list below).
2. Which trope or pattern it represents. The listed tropes are common examples \
for each genre, not an exhaustive closed set. If the excerpt contains a signal \
that clearly fits a genre's spirit but doesn't match any listed trope exactly, \
name the pattern yourself and still file the signal. Use your best judgment — \
the genre assignment must be exact, the trope name can be your own.
3. A tight one-sentence SIGNAL containing a verbatim quotation from the excerpt.

Genres and their representative tropes (not exhaustive):
{genre_block}

Output one block per signal:

GENRE: [exact genre name from the list]
TROPE: [listed trope name, or your own label if none fits]
SIGNAL: [one-sentence paraphrase with a verbatim quotation from the excerpt]
---

If no genre tropes are identifiable in this excerpt, output: NO SIGNALS
No text outside the structured blocks."""


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_signals(response: str, valid_genres: set[str]) -> list[TropeSignal]:
    if "NO SIGNALS" in response.upper() and "GENRE:" not in response.upper():
        return []
    signals: list[TropeSignal] = []
    for block in re.split(r"\n---+\n?", response):
        block = block.strip()
        if not block:
            continue
        genre_m  = re.search(r"^GENRE:\s*(.+)$",  block, re.MULTILINE)
        trope_m  = re.search(r"^TROPE:\s*(.+)$",  block, re.MULTILINE)
        signal_m = re.search(r"^SIGNAL:\s*(.+)$", block, re.MULTILINE)
        if not (genre_m and signal_m):
            continue
        raw_genre = genre_m.group(1).strip()
        raw_lc    = raw_genre.lower()
        matched   = next((g for g in valid_genres if g.lower() == raw_lc), None)
        if matched is None:
            matched = next(
                (g for g in valid_genres if g.lower() in raw_lc or raw_lc in g.lower()),
                None,
            )
        if matched is None:
            continue
        trope  = trope_m.group(1).strip() if trope_m else ""
        signal = signal_m.group(1).strip().strip("\"""'''").strip()
        signals.append(TropeSignal(genre=matched, trope=trope, signal=signal))
    return signals


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word-token sets (cheap fuzzy match)."""
    wa = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    wb = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _deduplicate(signals: list[TropeSignal]) -> list[TropeSignal]:
    out: list[TropeSignal] = []
    for s in signals:
        # Same genre + same trope name → check signal similarity
        is_dup = False
        for kept in out:
            if kept.genre != s.genre:
                continue
            # Exact prefix match (original check)
            if kept.signal[:70].lower().strip() == s.signal[:70].lower().strip():
                is_dup = True
                break
            # Same trope name + high token overlap → paraphrase duplicate
            if (kept.trope.lower() == s.trope.lower()
                    and _token_overlap(kept.signal, s.signal) >= 0.55):
                is_dup = True
                break
        if not is_dup:
            out.append(s)
    return out


def _build_results(signals: list[TropeSignal], genres: list[dict]) -> list[GenreResult]:
    by_genre: dict[str, list[TropeSignal]] = {g["name"]: [] for g in genres}
    for s in signals:
        if s.genre in by_genre:
            by_genre[s.genre].append(s)
    results: list[GenreResult] = []
    for genre_name, sigs in by_genre.items():
        if not sigs:
            continue
        results.append(GenreResult(
            genre=genre_name,
            total_signals=len(sigs),
            trope_signals=sigs,
        ))
    results.sort(key=lambda r: -r.total_signals)
    return results


# ── async pipeline ────────────────────────────────────────────────────────────

async def _analyze_async(
    text: str,
    model: str,
    on_progress,
    enabled_genres: list[str] | None = None,
    on_raw=None,
) -> tuple[list[GenreResult], TokenUsage]:
    all_genres   = load_genres()
    genres       = (
        [g for g in all_genres if g["name"] in enabled_genres]
        if enabled_genres is not None else all_genres
    ) or all_genres
    valid_genres = {g["name"] for g in genres}
    system       = _annotation_system(genres)
    chunks       = _chunk(text, ANNOTATION_CHUNK_CHARS)
    client       = anthropic.AsyncAnthropic(api_key=_get_api_key())
    sem          = asyncio.Semaphore(MAX_CONCURRENT)
    total        = len(chunks)
    done         = [0]
    usages: list[TokenUsage] = []

    async def _call(idx: int, chunk: str) -> list[TropeSignal]:
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
                on_progress("Detecting genre signals", done[0], total)
            if on_raw:
                on_raw("genre", idx, chunk, raw)
            sigs = _parse_signals(raw, valid_genres)
            for s in sigs:
                s.chunk_idx = idx
            return sigs

    results   = await asyncio.gather(*(_call(i, c) for i, c in enumerate(chunks)))
    all_sigs  = [s for chunk_sigs in results for s in chunk_sigs]
    all_sigs  = _deduplicate(all_sigs)
    return _build_results(all_sigs, genres), sum(usages, TokenUsage())


# ── public API ────────────────────────────────────────────────────────────────

def analyze_genres(
    filepath: str,
    model: str = DEFAULT_MODEL,
    on_progress=None,
    enabled_genres: list[str] | None = None,
    on_raw=None,
) -> tuple[list[GenreResult], TokenUsage]:
    """Detect genre tropes in a .docx file. Returns (results, usage)."""
    text = load_docx(filepath)
    if not text.strip():
        return [], TokenUsage()
    return asyncio.run(_analyze_async(text, model, on_progress, enabled_genres, on_raw))
