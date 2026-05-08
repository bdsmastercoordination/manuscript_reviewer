"""
Two-pass emotion analysis for Mirror.

Identical pipeline structure to character_analyzer but scores character
emotions (Joy, Sadness, Anger, Fear, Love, Shame, Disgust, Surprise,
Longing, Pride) instead of narrative archetypes.

Pass 1  Character discovery -- shared with character_analyzer.
Pass 2  Emotion annotation -- finds and scores emotional moments.
"""

import json
import asyncio
import re
from pathlib import Path

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx
from .character_analyzer import (
    RosterEntry, CharacterSheet, _Signal, _chunk,
    _aggregate_roster, _run_discovery,
    _parse_scored_signals, _build_scored_sheets, _run_annotation_pass,
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
    return _parse_scored_signals(response, roster, "EMOTIONS")

def _build_emotion_sheets(
    signals: list[_Signal],
    roster: list[RosterEntry],
    emotions: list[dict],
) -> list[CharacterSheet]:
    return _build_scored_sheets(signals, roster, emotions)

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
    return await _run_annotation_pass(
        chunks, _emotion_annotation_system(roster, emotions),
        _parse_emotion_signals, roster,
        client, model, sem, on_progress,
        phase="Annotating emotions",
        on_raw=on_raw, raw_phase="annotation",
    )

# ── async orchestration ───────────────────────────────────────────────────────

async def _analyze_async(
    text: str,
    model: str,
    roster_size: int,
    on_progress,
    synopsis: str,
    focal_characters: list[str] | None = None,
    enabled_emotions: list[str] | None = None,
    extra_emotions:   list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    all_emotions = load_emotions()
    if extra_emotions:
        all_emotions = all_emotions + extra_emotions
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
    extra_emotions:   list[dict] | None = None,
    on_raw=None,
) -> tuple[list[CharacterSheet], TokenUsage]:
    """
    Run emotion analysis on a .docx file.

    focal_characters  -- skip Pass 1 and use these names as the roster.
    enabled_emotions  -- restrict scoring to these emotion IDs (None = all).
    extra_emotions    -- additional emotion dicts (id, name, description) to include.
    on_progress(phase, current, total) -- called after each API call.
    """
    text = load_docx(filepath)
    if not text.strip():
        return [], TokenUsage()
    return asyncio.run(
        _analyze_async(
            text, model, roster_size, on_progress, synopsis,
            focal_characters, enabled_emotions, extra_emotions, on_raw,
        )
    )
