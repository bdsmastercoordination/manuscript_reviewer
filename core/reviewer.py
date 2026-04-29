import os
import asyncio
import re
from dataclasses import dataclass
from docx import Document
import anthropic
from .prompts import build_system_prompt, build_review_prompt, get_valid_types

REVIEWER_MODEL = "claude-haiku-4-5-20251001"  # alternatives: "claude-sonnet-4-6", "claude-opus-4-7"
CHUNK_CHARS      = 2000
MAX_CONCURRENT   = 3
OUTPUT_TOKEN_CAP = 4096

# Error types are split into three focused groups. Each chunk receives one API
# call per group so the model can concentrate on a narrower set of categories.
PASS_GROUPS: dict[str, frozenset[str]] = {
    "Mechanical": frozenset({
        "SPELLING", "PUNCTUATION", "WHITESPACE", "EM DASH",
        "GRAMMAR", "HYPHENATION", "NUMBER",
    }),
    "Style": frozenset({
        "REPETITION", "CUTTABLE MODIFIER", "REDUNDANT THAT",
        "WEAK VERB", "WORDY PHRASE", "VAGUE WORD", "REPETITIVE SYNTAX",
    }),
    "Craft": frozenset({
        "UNUSUAL WORD", "FILTER WORD", "THROAT-CLEARING",
        "PASSIVE VOICE", "TELLING EMOTION", "INCONSISTENT NAMING",
    }),
}

_async_client: anthropic.AsyncAnthropic | None = None


@dataclass
class Flag:
    type:  str
    text:  str
    issue: str
    tier:  str = "suggestion"


@dataclass
class TokenUsage:
    input:  int = 0
    output: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(self.input + other.input, self.output + other.output)


@dataclass
class ChunkResult:
    chunk:       int
    total:       int
    preview:     str
    flags:       list[Flag]
    pass1_usage: TokenUsage


def _get_api_key() -> str:
    def _looks_valid(k: str) -> bool:
        return bool(k) and k.startswith("sk-ant-")

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if _looks_valid(key):
        return key

    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            value, _ = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")
            value = value.strip()
            if _looks_valid(value):
                return value
    except Exception:
        pass

    raise EnvironmentError("ANTHROPIC_API_KEY not set or invalid")


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    return _async_client


def load_docx(path: str) -> str:
    with open(path, "rb") as f:
        doc = Document(f)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text[:CHUNK_CHARS]]


def parse_flags(response: str,
                enabled_types: set[str] | None = None) -> list[Flag]:
    valid_types = get_valid_types(enabled_types)
    seen: set[tuple[str, str]] = set()
    flags: list[Flag] = []

    blocks = re.split(r"\n---+\n?", response)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        text_m  = re.search(r'^TEXT:\s*"?(.+?)"?\s*$', block, re.MULTILINE)
        type_m  = re.search(r"^TYPE:\s*(.+)$",        block, re.MULTILINE)
        issue_m = re.search(r"^ISSUE:\s*(.+)$",        block, re.MULTILINE)
        if not (text_m and type_m):
            continue

        ftype = type_m.group(1).strip().upper()
        ftext = text_m.group(1).strip()

        if ftype not in valid_types:
            continue

        if not issue_m:
            continue

        dedup_key = (ftype, ftext[:20].lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        flags.append(Flag(
            type=ftype,
            text=ftext,
            issue=issue_m.group(1).strip(),
        ))
    return flags


async def _first_pass_async(chunk: str, model: str,
                             enabled_types: set[str] | None = None) -> tuple[list[Flag], TokenUsage]:
    for attempt in range(5):
        try:
            resp = await _get_async_client().messages.create(
                model=model,
                max_tokens=OUTPUT_TOKEN_CAP,
                temperature=0,
                system=build_system_prompt(enabled_types),
                messages=[{"role": "user", "content": build_review_prompt(chunk, enabled_types)}],
            )
            break
        except anthropic.RateLimitError:
            wait = 20 * (2 ** attempt)
            print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/5…", flush=True)
            await asyncio.sleep(wait)
    else:
        raise RuntimeError("Rate limit — all retries exhausted")
    usage = TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens)
    raw = resp.content[0].text.strip()

    if resp.stop_reason == "max_tokens":
        print(f"  [WARNING] output truncated at chunk (max_tokens reached)")

    if "NO ISSUES" in raw.upper() or "TYPE:" not in raw:
        return [], usage

    return parse_flags(raw, enabled_types), usage


async def _process_chunk_async(
    i: int, chunk: str, total: int, sem: asyncio.Semaphore,
    model: str, enabled_types: set[str] | None = None,
) -> ChunkResult | None:
    async with sem:
        all_flags: list[Flag] = []
        total_usage = TokenUsage()

        for group_types in PASS_GROUPS.values():
            # Intersect group with enabled_types (GUI checkbox filter)
            active = group_types if enabled_types is None else (
                group_types & {t.upper() for t in enabled_types}
            )
            if not active:
                continue
            flags, usage = await _first_pass_async(chunk, model, active)
            all_flags.extend(flags)
            total_usage = total_usage + usage

        if not all_flags:
            return None
        preview = chunk[:90].replace("\n", " ") + ("..." if len(chunk) > 90 else "")
        return ChunkResult(chunk=i + 1, total=total, preview=preview,
                           flags=all_flags, pass1_usage=total_usage)


async def _review_document_async(
    filepath: str, on_progress=None, model: str = REVIEWER_MODEL,
    enabled_types: set[str] | None = None,
) -> tuple[list[ChunkResult], TokenUsage]:
    text = load_docx(filepath)
    if not text.strip():
        raise ValueError("The document appears to be empty.")

    chunks = chunk_text(text)
    total  = len(chunks)
    sem    = asyncio.Semaphore(MAX_CONCURRENT)
    completed = [0]

    async def _tracked(i: int, chunk: str) -> ChunkResult | None:
        result = await _process_chunk_async(i, chunk, total, sem, model, enabled_types)
        completed[0] += 1
        if on_progress:
            on_progress(completed[0], total)
        return result

    raw_results = await asyncio.gather(*(_tracked(i, c) for i, c in enumerate(chunks)))

    results     = [r for r in raw_results if r is not None]
    total_usage = sum((r.pass1_usage for r in results), TokenUsage())
    return results, total_usage


def review_document(
    filepath: str, on_progress=None, model: str = REVIEWER_MODEL,
    enabled_types: set[str] | None = None,
) -> tuple[list[ChunkResult], TokenUsage]:
    return asyncio.run(_review_document_async(filepath, on_progress, model, enabled_types))
