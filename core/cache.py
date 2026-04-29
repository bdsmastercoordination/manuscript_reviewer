"""
Result cache for Mirror — persists copyedit and archetype results to JSON so
a previously-analyzed file can be reloaded without re-running the API.

Cache files live in  <project_root>/cache/
File names:  {stem}_{md5_8}.copyedit.json
             {stem}_{md5_8}.archetypes.json

The MD5 is computed from the docx file content, so the cache is automatically
invalidated when the file changes.
"""

import functools
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .reviewer import ChunkResult, Flag
from .character_analyzer import CharacterSheet
from .synopsis_analyzer import SynopsisResult, PlotBeat

CACHE_DIR = Path(__file__).parent.parent / "cache"


# ── key helpers ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _file_hash_cached(filepath: str, mtime: float) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:8]


def _file_hash(filepath: str) -> str:
    return _file_hash_cached(filepath, os.path.getmtime(filepath))


def _cache_path(filepath: str, kind: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    stem = Path(filepath).stem
    return CACHE_DIR / f"{stem}_{_file_hash(filepath)}.{kind}.json"


# ── copyedit ──────────────────────────────────────────────────────────────────

def _flag_to_dict(f: Flag) -> dict:
    return {"type": f.type, "text": f.text, "issue": f.issue, "tier": f.tier}


def _flag_from_dict(d: dict) -> Flag:
    return Flag(type=d["type"], text=d["text"], issue=d["issue"],
                tier=d.get("tier", "suggestion"))


def _chunk_to_dict(r: ChunkResult) -> dict:
    return {
        "chunk":   r.chunk,
        "total":   r.total,
        "preview": r.preview,
        "flags":   [_flag_to_dict(f) for f in r.flags],
    }


def _chunk_from_dict(d: dict) -> ChunkResult:
    from .reviewer import TokenUsage
    return ChunkResult(
        chunk=d["chunk"], total=d["total"], preview=d["preview"],
        flags=[_flag_from_dict(f) for f in d["flags"]],
        pass1_usage=TokenUsage(),
    )


def save_copyedit_cache(filepath: str, results: list[ChunkResult]) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results":     [_chunk_to_dict(r) for r in results],
    }
    _cache_path(filepath, "copyedit").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_copyedit_cache(filepath: str) -> dict | None:
    """Return parsed cache dict or None if no valid cache exists."""
    p = _cache_path(filepath, "copyedit")
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["results"] = [_chunk_from_dict(r) for r in payload["results"]]
        return payload
    except Exception:
        return None


# ── archetypes ────────────────────────────────────────────────────────────────

def _sheet_to_dict(s: CharacterSheet) -> dict:
    return {
        "name":                s.name,
        "aliases":             s.aliases,
        "synopsis":            s.synopsis,
        "top_archetypes":      [list(t) for t in s.top_archetypes],
        "evidence":            s.evidence,
        "total_signals":       s.total_signals,
        "chunks_seen":         s.chunks_seen,
        "signals_ordered":     s.signals_ordered,
        "neg_signals_ordered": s.neg_signals_ordered,
    }


def _sheet_from_dict(d: dict) -> CharacterSheet:
    return CharacterSheet(
        name=d["name"],
        aliases=d["aliases"],
        synopsis=d["synopsis"],
        top_archetypes=[tuple(t) for t in d["top_archetypes"]],
        evidence=d["evidence"],
        total_signals=d["total_signals"],
        chunks_seen=d["chunks_seen"],
        signals_ordered=d.get("signals_ordered", []),
        neg_signals_ordered=d.get("neg_signals_ordered", []),
    )


def save_arch_cache(filepath: str, sheets: list[CharacterSheet]) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sheets":      [_sheet_to_dict(s) for s in sheets],
    }
    _cache_path(filepath, "archetypes").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_arch_cache(filepath: str) -> dict | None:
    """Return parsed cache dict or None if no valid cache exists."""
    p = _cache_path(filepath, "archetypes")
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["sheets"] = [_sheet_from_dict(s) for s in payload["sheets"]]
        return payload
    except Exception:
        return None


# ── emotions ──────────────────────────────────────────────────────────────────

def save_emotion_cache(filepath: str, sheets: list[CharacterSheet]) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sheets":      [_sheet_to_dict(s) for s in sheets],
    }
    _cache_path(filepath, "emotions").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_emotion_cache(filepath: str) -> dict | None:
    """Return parsed cache dict or None if no valid cache exists."""
    p = _cache_path(filepath, "emotions")
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["sheets"] = [_sheet_from_dict(s) for s in payload["sheets"]]
        return payload
    except Exception:
        return None


# ── genres ────────────────────────────────────────────────────────────────────

def _genre_result_to_dict(r) -> dict:
    return {
        "genre":         r.genre,
        "total_signals": r.total_signals,
        "trope_signals": [
            {"genre": s.genre, "trope": s.trope, "signal": s.signal,
             "chunk_idx": s.chunk_idx}
            for s in r.trope_signals
        ],
    }


def _genre_result_from_dict(d: dict):
    from .genre_analyzer import TropeSignal, GenreResult
    return GenreResult(
        genre=d["genre"],
        total_signals=d["total_signals"],
        trope_signals=[TropeSignal(**s) for s in d["trope_signals"]],
    )


# ── synopsis ──────────────────────────────────────────────────────────────────

def save_synopsis_cache(filepath: str, result: SynopsisResult) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "logline":     result.logline,
        "synopsis":    result.synopsis,
        "beat_notes":  result.beat_notes,
        "plot_beats":  [
            {"term": b.term, "beat_idx": b.beat_idx,
             "summary": b.summary, "detail": b.detail}
            for b in result.plot_beats
        ],
    }
    _cache_path(filepath, "synopsis").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_synopsis_cache(filepath: str) -> dict | None:
    p = _cache_path(filepath, "synopsis")
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        result = SynopsisResult(
            logline=d.get("logline", ""),
            synopsis=d.get("synopsis", ""),
            beat_notes=d.get("beat_notes", []),
            plot_beats=[PlotBeat(**b) for b in d.get("plot_beats", [])],
        )
        return {"analyzed_at": d["analyzed_at"], "result": result}
    except Exception:
        return None


def save_genre_cache(filepath: str, results: list) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results":     [_genre_result_to_dict(r) for r in results],
    }
    _cache_path(filepath, "genres").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_genre_cache(filepath: str) -> dict | None:
    """Return parsed cache dict or None if no valid cache exists."""
    p = _cache_path(filepath, "genres")
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["results"] = [_genre_result_from_dict(r) for r in payload["results"]]
        return payload
    except Exception:
        return None


# ── opening ───────────────────────────────────────────────────────────────────

def save_opening_cache(filepath: str, result) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": [
            {"item": it.item, "guess": it.guess, "confidence": it.confidence}
            for it in result.items
        ],
    }
    _cache_path(filepath, "opening").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_opening_cache(filepath: str) -> dict | None:
    p = _cache_path(filepath, "opening")
    if not p.exists():
        return None
    try:
        from .opening_analyzer import OpeningCheckItem, OpeningResult
        d      = json.loads(p.read_text(encoding="utf-8"))
        result = OpeningResult(
            items=[OpeningCheckItem(**it) for it in d["items"]]
        )
        return {"analyzed_at": d["analyzed_at"], "result": result}
    except Exception:
        return None


# ── arcs ──────────────────────────────────────────────────────────────────────

def save_arcs_cache(filepath: str, result) -> None:
    payload = {
        "version":     1,
        "filepath":    filepath,
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "structural":  [{"name": a.name, "start": a.start, "end": a.end}
                        for a in result.structural],
        "character":   [{"name": a.name, "start": a.start, "end": a.end}
                        for a in result.character],
    }
    _cache_path(filepath, "arcs").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_arcs_cache(filepath: str) -> dict | None:
    p = _cache_path(filepath, "arcs")
    if not p.exists():
        return None
    try:
        from .arc_analyzer import ArcResult, ArcsResult
        d      = json.loads(p.read_text(encoding="utf-8"))
        result = ArcsResult(
            structural=[ArcResult(**a) for a in d["structural"]],
            character=[ArcResult(**a) for a in d["character"]],
        )
        return {"analyzed_at": d["analyzed_at"], "result": result}
    except Exception:
        return None
