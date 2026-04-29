"""
Held-out test set evaluator. Do NOT use to tune prompts — only for final validation.
test_set.docx is a hand-maintained fixture; this script never writes to it.
Run: python eval_test.py
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reviewer import review_document, ChunkResult, TokenUsage

TEST_DOC_PATH = Path(__file__).parent / "test_set.docx"

# ---------------------------------------------------------------------------
# Known errors in test_set.docx (16 paragraphs, hand-crafted fixture).
# ---------------------------------------------------------------------------
KNOWN_ERRORS = [
    # ── para 1: spelling ──────────────────────────────────────────────────
    {"id": "spell-1",  "description": "'neccessary' -> 'necessary'",    "keywords": ["neccessary",  "necessary"]},
    {"id": "spell-2",  "description": "'independant' -> 'independent'", "keywords": ["independant", "independent"]},
    {"id": "spell-3",  "description": "'reccomend' -> 'recommend'",     "keywords": ["reccomend",   "recommend"]},
    # ── para 2: punctuation ───────────────────────────────────────────────
    {"id": "punct-1",  "description": "Missing comma after 'Although he had waited'", "keywords": ["although", "comma", "introductory"]},
    {"id": "punct-2",  "description": "Missing apostrophe: 'shouldnt'", "keywords": ["shouldnt",  "shouldn't"]},
    {"id": "punct-3",  "description": "Missing apostrophe: 'doesnt'",   "keywords": ["doesnt",    "doesn't"]},
    # ── para 3: whitespace ────────────────────────────────────────────────
    {"id": "ws-1",     "description": "Double space before 'confirmed'", "keywords": ["confirmed", "double space", "extra space", "whitespace"]},
    {"id": "ws-2",     "description": "Double space before 'standing'",  "keywords": ["standing",  "double space", "extra space", "whitespace"]},
    # ── para 4: em dash ───────────────────────────────────────────────────
    {"id": "em-1",     "description": "'--' after 'admitted'",  "keywords": ["admitted",  "em dash", "double hyphen"]},
    {"id": "em-2",     "description": "'--' after 'notebook'",  "keywords": ["notebook",  "em dash", "double hyphen"]},
    # ── para 5: repetition ────────────────────────────────────────────────
    {"id": "rep-1",    "description": "'turned' repeated four times", "keywords": ["turned"]},
    # ── para 6: grammar ───────────────────────────────────────────────────
    {"id": "gram-1",   "description": "'married with' -> 'married to'", "keywords": ["married with",  "married to"]},
    {"id": "gram-2",   "description": "'afraid from' -> 'afraid of'",   "keywords": ["afraid from",   "afraid of"]},
    {"id": "gram-3",   "description": "'insist to' -> 'insist on'",     "keywords": ["insist to",     "insist on"]},
    # ── para 8: hyphenation ───────────────────────────────────────────────
    {"id": "hyph-1",   "description": "'well timed' missing hyphen",    "keywords": ["well timed",  "well-timed"]},
    {"id": "hyph-2",   "description": "'long running' missing hyphen",  "keywords": ["long running", "long-running"]},
    {"id": "hyph-3",   "description": "'high stakes' missing hyphen",   "keywords": ["high stakes", "high-stakes"]},
    # ── para 8: number ────────────────────────────────────────────────────
    {"id": "num-1",    "description": "'4' should be 'four'",  "keywords": ["4 hours",   "four hours"]},
    {"id": "num-2",    "description": "'9' should be 'nine'",  "keywords": ["9 of the",  "nine of the"]},
    {"id": "num-3",    "description": "'5' should be 'five'",  "keywords": ["5 that",    "five that"]},
    # ── para 9: cuttable modifier ─────────────────────────────────────────
    {"id": "cut-1",    "description": "cuttable 'very' in 'very uncertain'",   "keywords": ["very uncertain"]},
    {"id": "cut-2",    "description": "cuttable 'really' in 'really difficult'", "keywords": ["really difficult"]},
    {"id": "cut-3",    "description": "cuttable 'quite' in 'quite sure'",      "keywords": ["quite sure"]},
    # ── para 9: redundant that ────────────────────────────────────────────
    {"id": "rt-1",     "description": "redundant 'that' after 'thought'",  "keywords": ["thought that the meeting", "thought that"]},
    {"id": "rt-2",     "description": "redundant 'that' after 'knew'",     "keywords": ["knew that appearances",   "knew that"]},
    # ── para 10: wordy phrase ─────────────────────────────────────────────
    {"id": "wp-1",     "description": "'was able to confirm' is wordy",    "keywords": ["was able to confirm",    "able to"]},
    {"id": "wp-2",     "description": "'was able to determine' is wordy",  "keywords": ["was able to determine",  "able to"]},
    # ── para 10: weak verb / vague word ───────────────────────────────────
    {"id": "wv-1",     "description": "'went into the back room' is a weak verb",  "keywords": ["went into", "entered"]},
    {"id": "vw-1",     "description": "'the thing she found' is vague",            "keywords": ["the thing she found", "thing she"]},
    {"id": "wv-2",     "description": "'got the ledger' is a weak verb",           "keywords": ["got the ledger", "retrieved", "fetched"]},
    # ── para 11: repetitive syntax ────────────────────────────────────────
    {"id": "rs-1",     "description": "He placed / He checked / He addressed / He sealed: same structure ×4", "keywords": ["he placed", "he checked", "he addressed", "repetitive syntax", "repetitive structure"]},
    # ── para 11: telling emotion ──────────────────────────────────────────
    {"id": "te-1",     "description": "telling emotion: 'felt guilty'",    "keywords": ["felt guilty",   "guilty"]},
    {"id": "te-2",     "description": "telling emotion: 'felt excited'",   "keywords": ["felt excited",  "excited"]},
    # ── para 12: filter word ──────────────────────────────────────────────
    {"id": "fw-1",     "description": "filter word: 'heard that the corridor'", "keywords": ["heard that the corridor", "heard that"]},
    {"id": "fw-2",     "description": "filter word: 'felt that something'",     "keywords": ["felt that something",     "felt that"]},
    # ── para 12: throat-clearing ──────────────────────────────────────────
    {"id": "tc-1",     "description": "throat-clearing: 'It was not until...that she realised'", "keywords": ["it was not until", "not until she"]},
    {"id": "tc-2",     "description": "throat-clearing: 'What struck her was'",                  "keywords": ["what struck her was"]},
    # ── para 12: passive voice ────────────────────────────────────────────
    {"id": "pv-1",     "description": "passive: 'had been marked by someone'", "keywords": ["been marked by", "marked by someone"]},
    # ── para 12: cliché metaphor ──────────────────────────────────────────
    {"id": "clm-1",    "description": "'tip of the iceberg' is a cliché",  "keywords": ["tip of the iceberg"]},
    {"id": "clm-2",    "description": "'twist of fate' is a cliché",       "keywords": ["twist of fate"]},
    # ── para 13: grammar ──────────────────────────────────────────────────
    {"id": "gram-4",   "description": "'differed to' -> 'differed from'",  "keywords": ["differed to",  "differed from"]},
    {"id": "gram-5",   "description": "'resulted to' -> 'resulted in'",    "keywords": ["resulted to",  "resulted in"]},
    {"id": "gram-6",   "description": "'familiar of' -> 'familiar with'",  "keywords": ["familiar of",  "familiar with"]},
    # ── para 13: spelling ─────────────────────────────────────────────────
    {"id": "spell-4",  "description": "'priviledge' -> 'privilege'",   "keywords": ["priviledge",  "privilege"]},
    {"id": "spell-5",  "description": "'occured' -> 'occurred'",       "keywords": ["occured",     "occurred"]},
    {"id": "spell-6",  "description": "'manuever' -> 'maneuver'",      "keywords": ["manuever",    "maneuver"]},
    # ── para 15: inconsistent naming ──────────────────────────────────────
    {"id": "naming-1", "description": "'Oskar' vs 'Oscar' inconsistency",     "keywords": ["oskar", "oscar",   "inconsistent"]},
    {"id": "naming-2", "description": "'Carraway' vs 'Caraway' inconsistency", "keywords": ["carraway", "caraway", "inconsistent"]},
    # ── para 16: language (British spellings) ─────────────────────────────
    {"id": "lang-1",   "description": "'realised' should be 'realized'",    "keywords": ["realised",   "british spelling", "spelling convention"]},
    {"id": "lang-2",   "description": "'colour' should be 'color'",         "keywords": ["colour",     "british spelling", "spelling convention"]},
    {"id": "lang-3",   "description": "'organised' should be 'organized'",  "keywords": ["organised",  "british spelling", "spelling convention"]},
    {"id": "lang-4",   "description": "'recognised' should be 'recognized'","keywords": ["recognised", "british spelling", "spelling convention"]},
    {"id": "lang-5",   "description": "'travelling' should be 'traveling'", "keywords": ["travelling", "british spelling", "spelling convention"]},
]

CLEAN_CHUNKS = {4}  # Para 7 (the long negotiation/canal walk section) forms its own clean chunk


def _progress(current: int, total: int) -> None:
    print(f"  chunk {current}/{total}...", flush=True)


def score(results: list[ChunkResult], usage: TokenUsage) -> None:
    all_issues = "\n".join(
        " ".join([f.type, f.text, f.issue]) for r in results for f in r.flags
    ).lower()
    flagged_chunks  = {r.chunk for r in results}
    false_positives = flagged_chunks & CLEAN_CHUNKS

    divider = "=" * 62
    print(f"\n{divider}\nSCORE\n{divider}")

    detected = 0
    for err in KNOWN_ERRORS:
        hit = any(kw.lower() in all_issues for kw in err["keywords"])
        if hit:
            detected += 1
        print(f"  [{'PASS' if hit else 'FAIL'}]  {err['id']:10s}  {err['description']}")

    total = len(KNOWN_ERRORS)
    pct = round(100 * detected / total)
    print(f"\n  Recall:  {detected}/{total} ({pct}%)")

    if false_positives:
        print(f"  FALSE POSITIVES: model flagged clean chunk(s) {sorted(false_positives)}")
    else:
        print("  False positives: none")

    print(f"\n  Tokens:  {usage.input:,} in / {usage.output:,} out")
    print(divider + "\n")


def main() -> None:
    if not TEST_DOC_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_DOC_PATH}")
    print("Running reviewer on held-out test set...")
    results, usage = review_document(str(TEST_DOC_PATH), on_progress=_progress)
    print()

    if results:
        print("RAW OUTPUT\n" + "-" * 62)
        for r in results:
            print(f"\n-- Chunk {r.chunk}/{r.total} --\n   {r.preview}")
            for f in r.flags:
                print(f"  [{f.type}]")
                print(f"    TEXT:  {f.text}")
                print(f"    ISSUE: {f.issue}")

    score(results, usage)


if __name__ == "__main__":
    main()
