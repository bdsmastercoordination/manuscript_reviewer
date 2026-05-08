"""
Two-call arc analysis for Mirror — runs after synopsis is complete.

Pass 1  Structural arcs — which plot-level arc types apply.
Pass 2  Character arcs  — start/end arc for each main character.

Both calls receive the logline + synopsis + labeled plot beats as context.
Output per arc: name/character, one-phrase start, one-phrase end.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from .reviewer import _get_api_key, TokenUsage
from .character_analyzer import DEFAULT_MODEL, MAX_CONCURRENT
from .synopsis_analyzer import SynopsisResult


STRUCTURAL_ARCS_PATH = Path(__file__).parent.parent / "config" / "structural_arcs.jsonl"


def load_structural_arcs() -> list[dict]:
    with open(STRUCTURAL_ARCS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ArcResult:
    name:  str
    start: str
    end:   str


@dataclass
class ArcsResult:
    structural: list[ArcResult] = field(default_factory=list)
    character:  list[ArcResult] = field(default_factory=list)


# ── context builder ───────────────────────────────────────────────────────────

def _build_context(result: SynopsisResult) -> str:
    parts = []
    if result.logline:
        parts.append(f"Logline: {result.logline}")
    if result.synopsis:
        parts.append(f"Synopsis: {result.synopsis}")
    if result.plot_beats:
        beats_text = "\n".join(
            f"  {b.term}: {b.summary}" for b in result.plot_beats
        )
        parts.append(f"Plot beats:\n{beats_text}")
    return "\n\n".join(parts)


# ── parser ────────────────────────────────────────────────────────────────────

def _parse_arcs(text: str) -> list[ArcResult]:
    if "NONE" in text.upper() and "NAME:" not in text.upper():
        return []
    results: list[ArcResult] = []
    for block in re.split(r"\n---+\n?", text):
        block = block.strip()
        if not block:
            continue
        name_m  = re.search(r"^NAME:\s*(.+)$",  block, re.MULTILINE)
        start_m = re.search(r"^START:\s*(.+)$", block, re.MULTILINE)
        end_m   = re.search(r"^END:\s*(.+)$",   block, re.MULTILINE)
        if not (name_m and start_m and end_m):
            continue
        results.append(ArcResult(
            name=name_m.group(1).strip(),
            start=start_m.group(1).strip(),
            end=end_m.group(1).strip(),
        ))
    seen = set()
    return [r for r in results if not (r.name in seen or seen.add(r.name))]


# ── pass 1: structural arcs ───────────────────────────────────────────────────

_STRUCTURAL_SYSTEM = """\
You are a narrative analyst. Based on the story context below, identify which \
of the listed structural arc types are clearly present in this story.

For each matching arc, output:

NAME: [arc type name exactly as listed]
START: [under 8 words: the opening situation for this arc]
END: [under 8 words: how this arc resolves or concludes]
---

Only include arcs with a clear, sustained presence. Omit arcs that are \
absent or only barely suggested. If none apply, output: NONE

No text outside the structured blocks."""

_STRUCTURAL_USER = """\
Structural arc types to identify (only use names from this list):
{arc_list}

Story context:
{context}"""


async def _run_structural(
    context: str,
    enabled_arcs: list[dict],
    client: anthropic.AsyncAnthropic,
    model: str,
    on_progress,
) -> tuple[list[ArcResult], TokenUsage]:
    arc_list = "\n".join(
        f"  - {a['name']}: {a['description']}" for a in enabled_arcs
    )
    for attempt in range(5):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=_STRUCTURAL_SYSTEM,
                messages=[{"role": "user",
                           "content": _STRUCTURAL_USER.format(
                               arc_list=arc_list, context=context)}],
            )
            break
        except anthropic.RateLimitError:
            await asyncio.sleep(20 * (2 ** attempt))
    else:
        return [], TokenUsage()

    if on_progress:
        on_progress("Identifying structural arcs…", 1, 2)
    usage  = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    arcs   = _parse_arcs(resp.content[0].text.strip())
    return arcs, usage


# ── pass 2: character arcs ────────────────────────────────────────────────────

_CHARACTER_SYSTEM = """\
You are a narrative analyst. For each listed character, describe their \
personal arc based on the story context below. Be extremely brief.

For each character, output:

NAME: [character name]
START: [under 8 words: who they are or where they stand at the opening]
END: [under 8 words: who they become or where they end up]
---

Only include characters who have a discernible arc. \
If a character has no meaningful arc, omit them. \
No text outside the structured blocks."""

_CHARACTER_USER = """\
Characters:
{characters}

Story context:
{context}"""


async def _run_character(
    context: str,
    character_names: list[str],
    client: anthropic.AsyncAnthropic,
    model: str,
    on_progress,
) -> tuple[list[ArcResult], TokenUsage]:
    if not character_names:
        return [], TokenUsage()
    chars = "\n".join(f"  - {n}" for n in character_names)
    for attempt in range(5):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=_CHARACTER_SYSTEM,
                messages=[{"role": "user",
                           "content": _CHARACTER_USER.format(
                               characters=chars, context=context)}],
            )
            break
        except anthropic.RateLimitError:
            await asyncio.sleep(20 * (2 ** attempt))
    else:
        return [], TokenUsage()

    if on_progress:
        on_progress("Mapping character arcs…", 2, 2)
    usage  = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    arcs   = _parse_arcs(resp.content[0].text.strip())
    return arcs, usage


# ── orchestration ─────────────────────────────────────────────────────────────

async def _analyze_async(
    synopsis_result: SynopsisResult,
    model: str,
    enabled_structural: list[dict] | None,
    include_character: bool,
    character_names: list[str],
    on_progress,
) -> tuple[ArcsResult, TokenUsage]:
    all_arcs  = load_structural_arcs()
    s_arcs    = enabled_structural if enabled_structural is not None else all_arcs
    context   = _build_context(synopsis_result)
    client    = anthropic.AsyncAnthropic(api_key=_get_api_key())

    structural, u1 = await _run_structural(context, s_arcs, client, model, on_progress)

    if include_character and character_names:
        character, u2 = await _run_character(context, character_names, client, model, on_progress)
    else:
        character, u2 = [], TokenUsage()

    return ArcsResult(structural=structural, character=character), u1 + u2


def analyze_arcs(
    synopsis_result: SynopsisResult,
    model: str = DEFAULT_MODEL,
    enabled_structural: list[dict] | None = None,
    include_character: bool = True,
    character_names: list[str] | None = None,
    on_progress=None,
) -> tuple[ArcsResult, TokenUsage]:
    """Identify structural and character arcs from a completed SynopsisResult."""
    return asyncio.run(_analyze_async(
        synopsis_result, model,
        enabled_structural, include_character,
        character_names or [],
        on_progress,
    ))
