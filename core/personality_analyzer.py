"""
Two-pass Big Five personality analysis for Mirror.

Identical pipeline structure to emotion_analyzer but scores character
personality traits (Openness, Conscientiousness, Extraversion, Agreeableness,
Neuroticism) instead of emotions.

Pass 1  Character discovery -- shared with character_analyzer.
Pass 2  Trait annotation -- finds and scores Big Five trait expressions.
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
    return _parse_scored_signals(response, roster, "TRAITS")

def _build_personality_sheets(
    signals: list[_Signal],
    roster: list[RosterEntry],
    traits: list[dict],
) -> list[CharacterSheet]:
    return _build_scored_sheets(signals, roster, traits, bipolar=True, threshold=0.1)

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
    return await _run_annotation_pass(
        chunks, _trait_annotation_system(roster, traits),
        _parse_trait_signals, roster,
        client, model, sem, on_progress,
        phase="Annotating personality",
        on_raw=on_raw, raw_phase="personality",
    )

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
