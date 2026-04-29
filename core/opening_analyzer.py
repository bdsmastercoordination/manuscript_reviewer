"""
Single-pass opening analysis for Mirror.

Reads the first ~1500 words and returns a reader's first impressions
across seven checklist elements.
"""

import asyncio
import re
from dataclasses import dataclass, field

import anthropic

from .reviewer import _get_api_key, TokenUsage, load_docx
from .character_analyzer import DEFAULT_MODEL


OPENING_WORDS = 1500

CHECKLIST_ITEMS = [
    "Hook",
    "Plot promise",
    "Conflict type",
    "Genre promise",
    "Tone/style promise",
    "Protagonist character",
    "Time and place",
    "Ending",
    "Target group",
]

# Three-level confidence: hug = clear, neutral = somewhat, confused = unclear
CONFIDENCE_EMOJI = {
    "hug":      "\U0001f917",   # 🤗
    "neutral":  "\U0001f610",   # 😐
    "confused": "\U0001f615",   # 😕
}


@dataclass
class OpeningCheckItem:
    item:       str
    guess:      str
    confidence: str   # "hug" | "neutral" | "confused"


@dataclass
class OpeningResult:
    items: list[OpeningCheckItem] = field(default_factory=list)


def _first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


_SYSTEM = """\
You are an editor jotting quick notes after reading the opening pages of a manuscript. \
Take guesses. Be terse. No evaluative language -- do not say whether the writing \
is good, clear, effective, or well-done. Do not reference yourself. \
Do not summarise or echo the text. Just state what you think is going on \
and what you expect, as brief working notes.

Tone: clipped, neutral, third-person or impersonal. Two short sentences maximum. \
Examples of the right register:
  "Domestic drama, probably a marriage unravelling."
  "Some kind of loss. Guilt driving it forward."
  "Could be literary fiction -- slow, interior, European feel."
  "No sense of time period yet."
  "Adult literary fiction. More specifically, readers who enjoy quiet, \
interior Scandinavian drama -- Knausgard territory."

For "Target group": give one broad audience category, then one more specific \
niche within that -- both in a single sentence. Example: "Adult literary fiction. \
More specifically, readers who enjoy quiet, interior Scandinavian drama."

For each other item, give your working guess and a confidence level:
  hug      = came through clearly
  neutral  = possible but uncertain
  confused = no real signal

Output one block per item:

ITEM: [item name]
GUESS: [working guess -- 1-2 sentences, clipped and impersonal, no self-reference, no praise]
CONFIDENCE: [hug | neutral | confused]
---

Items to assess:
{items}

No text outside the structured blocks."""


def _parse_result(text: str, enabled_items: list[str]) -> OpeningResult:
    items: list[OpeningCheckItem] = []
    for block in re.split(r"\n---+\n?", text):
        block = block.strip()
        if not block:
            continue
        item_m  = re.search(r"^ITEM:\s*(.+)$",  block, re.MULTILINE)
        guess_m = re.search(r"^GUESS:\s*(.+?)(?=\nCONFIDENCE:|$)", block,
                             re.MULTILINE | re.DOTALL)
        conf_m  = re.search(r"^CONFIDENCE:\s*(\w+)", block, re.MULTILINE)
        if not (item_m and guess_m):
            continue
        raw_name  = item_m.group(1).strip()
        canonical = next(
            (ei for ei in enabled_items
             if ei.lower() in raw_name.lower() or raw_name.lower() in ei.lower()),
            raw_name,
        )
        guess = guess_m.group(1).strip()
        raw_conf  = conf_m.group(1).lower() if conf_m else "neutral"
        confidence = raw_conf if raw_conf in CONFIDENCE_EMOJI else "neutral"
        items.append(OpeningCheckItem(item=canonical, guess=guess, confidence=confidence))
    return OpeningResult(items=items)


async def _analyze_async(
    text: str,
    model: str,
    enabled_items: list[str],
    on_progress,
) -> tuple[OpeningResult, TokenUsage]:
    excerpt    = _first_words(text, OPENING_WORDS)
    items_list = "\n".join(f"  - {it}" for it in enabled_items)
    system     = _SYSTEM.format(items=items_list)

    client = anthropic.AsyncAnthropic(api_key=_get_api_key())

    for attempt in range(5):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=system,
                messages=[{"role": "user",
                           "content": f"Opening excerpt:\n\n{excerpt}"}],
            )
            break
        except anthropic.RateLimitError:
            await asyncio.sleep(20 * (2 ** attempt))
    else:
        return OpeningResult(), TokenUsage()

    if on_progress:
        on_progress("Reading opening…", 1, 1)

    usage  = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    result = _parse_result(resp.content[0].text.strip(), enabled_items)
    return result, usage


def analyze_opening(
    filepath: str,
    model: str = DEFAULT_MODEL,
    enabled_items: list[str] | None = None,
    on_progress=None,
) -> tuple[OpeningResult, TokenUsage]:
    """Analyze the opening ~1500 words of a .docx manuscript."""
    text = load_docx(filepath)
    if not text.strip():
        return OpeningResult(), TokenUsage()
    items = enabled_items if enabled_items else CHECKLIST_ITEMS
    return asyncio.run(_analyze_async(text, model, items, on_progress))
