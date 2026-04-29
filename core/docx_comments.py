"""
Write editorial flags as Word review comments into a copy of the source docx.
Each flag is anchored to the exact quoted text span when found; falls back to
paragraph-level anchoring only when every significant word from the flag text
can be located in the paragraph. Skips rather than misplaces if no match.
"""

import copy
import shutil
from datetime import datetime, timezone
from lxml import etree
from docx import Document
from docx.opc.part import Part
from docx.opc.packuri import PackURI

W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
COMMENTS_URI = "/word/comments.xml"
COMMENTS_CT  = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
COMMENTS_RT  = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
AUTHOR   = "Mirror"
INITIALS = "MR"
DATE     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Normalization ─────────────────────────────────────────────────────────────
# 1:1 character substitutions — len(result) == len(input), so char positions
# in the normalized string correspond exactly to positions in the original.
_NORM_TABLE = str.maketrans({
    '\u201c': '"',  # left double quotation mark
    '\u201d': '"',  # right double quotation mark
    '\u2018': "'",  # left single quotation mark
    '\u2019': "'",  # right single quotation mark
    '\u2026': '.',  # horizontal ellipsis → single dot
    '\u2014': '-',  # em dash → hyphen
    '\u2013': '-',  # en dash → hyphen
    '\u00a0': ' ',  # non-breaking space → space
})

_STOP = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
         'for', 'of', 'with', 'by', 'from', 'is', 'was', 'he', 'she',
         'it', 'they', 'we', 'i', 'his', 'her', 'that', 'this'}


def _norm(text: str) -> str:
    """1:1 typographic normalization — positions preserved."""
    return text.translate(_NORM_TABLE)


def _clean_needle(raw: str) -> str:
    """Strip outer quotes and normalize typographic chars in the search needle."""
    s = raw.strip().strip('"\'').strip('\u201c\u201d\u2018\u2019').strip()
    s = _norm(s)
    s = s.replace('--', '-')  # double-hyphen (plain-text em dash) → single hyphen
    return ' '.join(s.split())


def _find_span(full: str, raw_needle: str) -> tuple[int, int]:
    """
    Locate raw_needle inside full. Returns (start, end) in `full`, or (-1, -1).

    Strategy:
      1. Try the full cleaned needle.
      2. Progressively drop words from the end until ≥ 4 words remain.
    Both full and needle are normalized with _norm before comparison so
    typographic characters (curly quotes, em dashes …) cannot cause mismatches.
    Positions returned are valid indices into the original `full` string because
    the normalization is 1:1.
    """
    needle = _clean_needle(raw_needle)
    full_n = _norm(full)
    words  = needle.split()

    # Build candidate anchors: full phrase → progressively shorter
    candidates = [needle]
    for n in range(len(words) - 1, 3, -1):  # stops at 4 words
        candidates.append(' '.join(words[:n]))

    for cand in candidates:
        idx = full_n.lower().find(cand.lower())
        if idx >= 0:
            return idx, idx + len(cand)

    return -1, -1


def _sig_words(text: str) -> list[str]:
    """Significant words (> 3 chars, not stop words) for fallback matching."""
    return [w for w in _norm(text).lower().split()
            if len(w) > 3 and w.strip("\"'.,!?;:-") not in _STOP]


# ── XML builders ─────────────────────────────────────────────────────────────

def _q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def _make_comment_xml(comments: list[tuple[int, str]]) -> bytes:
    root = etree.Element(_q("comments"), nsmap={"w": W})
    for cid, body in comments:
        c = etree.SubElement(root, _q("comment"))
        c.set(_q("id"), str(cid))
        c.set(_q("author"), AUTHOR)
        c.set(_q("date"), DATE)
        c.set(_q("initials"), INITIALS)
        p = etree.SubElement(c, _q("p"))
        r = etree.SubElement(p, _q("r"))
        t = etree.SubElement(r, _q("t"))
        t.text = body
        t.set(f"{{{XML}}}space", "preserve")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _comment_range_start(cid: int) -> etree._Element:
    el = etree.Element(_q("commentRangeStart"))
    el.set(_q("id"), str(cid))
    return el


def _comment_range_end(cid: int) -> etree._Element:
    el = etree.Element(_q("commentRangeEnd"))
    el.set(_q("id"), str(cid))
    return el


def _comment_ref_run(cid: int) -> etree._Element:
    r = etree.Element(_q("r"))
    rpr = etree.SubElement(r, _q("rPr"))
    style = etree.SubElement(rpr, _q("rStyle"))
    style.set(_q("val"), "CommentReference")
    ref = etree.SubElement(r, _q("commentReference"))
    ref.set(_q("id"), str(cid))
    return r


# ── Run helpers ───────────────────────────────────────────────────────────────

def _para_runs(para_el: etree._Element) -> list[etree._Element]:
    runs = []
    for child in para_el:
        if child.tag == _q("r"):
            runs.append(child)
        elif child.tag == _q("hyperlink"):
            runs.extend(c for c in child if c.tag == _q("r"))
    return runs


def _run_text(run_el: etree._Element) -> str:
    t = run_el.find(_q("t"))
    return t.text if t is not None and t.text else ""


def _set_run_text(run_el: etree._Element, text: str) -> None:
    t = run_el.find(_q("t"))
    if t is None:
        t = etree.SubElement(run_el, _q("t"))
    t.text = text
    if text and (text[0] == " " or text[-1] == " "):
        t.set(f"{{{XML}}}space", "preserve")
    elif f"{{{XML}}}space" in t.attrib:
        del t.attrib[f"{{{XML}}}space"]


def _split_run(run_el: etree._Element, offset: int) -> etree._Element:
    """
    Split run_el at char offset. run_el keeps text[:offset].
    Returns the new second-half run (text[offset:]), inserted immediately after.
    """
    text   = _run_text(run_el)
    second = copy.deepcopy(run_el)
    _set_run_text(run_el,  text[:offset])
    _set_run_text(second, text[offset:])
    run_el.addnext(second)
    return second


# ── Core anchoring ────────────────────────────────────────────────────────────

def _insert_markers_around(start_el: etree._Element,
                            end_el:   etree._Element,
                            cid: int) -> None:
    """
    Insert comment markers so the range covers start_el … end_el inclusive.
    Word requires:  commentRangeStart | … runs … | commentRangeEnd | commentRef
    """
    start_el.addprevious(_comment_range_start(cid))
    # addnext inserts immediately after; calling twice pushes first one right:
    #   end_el | commentRef  →  end_el | commentRangeEnd | commentRef
    end_el.addnext(_comment_ref_run(cid))
    end_el.addnext(_comment_range_end(cid))


def _anchor_comment(para_el: etree._Element, search: str, cid: int) -> bool:
    """
    Find `search` text inside para_el and wrap it with comment markers.
    Returns True on success.
    """
    runs = _para_runs(para_el)
    if not runs:
        return False

    # Build per-run char spans over the concatenated paragraph text
    spans: list[tuple[etree._Element, int, int]] = []
    pos = 0
    for r in runs:
        t = _run_text(r)
        spans.append((r, pos, pos + len(t)))
        pos += len(t)
    full = "".join(_run_text(r) for r in runs)

    start_char, end_char = _find_span(full, search)
    if start_char < 0:
        return False

    # Which run contains the start / end character?
    start_run_idx = next(
        (i for i, (_, rs, re) in enumerate(spans) if re > start_char), None
    )
    end_run_idx = next(
        (i for i, (_, rs, re) in enumerate(spans) if re >= end_char), None
    )
    if start_run_idx is None or end_run_idx is None:
        return False

    if start_run_idx == end_run_idx:
        # ── Same run: split into up to three pieces ───────────────────────────
        run_el, run_s, _ = spans[start_run_idx]
        start_off = start_char - run_s
        end_off   = end_char   - run_s
        run_len   = len(_run_text(run_el))

        # 1. Split off the tail (text after the target span) if it exists
        if end_off < run_len:
            _split_run(run_el, end_off)
            # run_el is now text[:end_off]

        # 2. Split off the head (text before the target span) if it exists
        if start_off > 0:
            target_el = _split_run(run_el, start_off)
            # run_el is now text[:start_off]; target_el is text[start_off:end_off]
        else:
            target_el = run_el  # whole (possibly truncated) run is the target

        _insert_markers_around(target_el, target_el, cid)

    else:
        # ── Different runs: split end run first, then start run ───────────────
        end_run_el, end_s, _ = spans[end_run_idx]
        end_off = end_char - end_s
        if 0 < end_off < len(_run_text(end_run_el)):
            _split_run(end_run_el, end_off)
            # end_run_el is now the portion we want inside the range

        start_run_el, start_s, _ = spans[start_run_idx]
        start_off = start_char - start_s
        if start_off > 0:
            start_run_el = _split_run(start_run_el, start_off)
            # start_run_el is now the second half (where the anchor begins)

        _insert_markers_around(start_run_el, end_run_el, cid)

    return True


# ── Public API ────────────────────────────────────────────────────────────────

def annotate(source_path: str, output_path: str, results) -> int:
    """
    Write a copy of source_path to output_path with one Word comment per flag.
    Returns the number of comments successfully anchored.
    """
    shutil.copy2(source_path, output_path)
    doc = Document(output_path)

    comment_data: list[tuple[int, str]] = []
    anchors: list[tuple[str, int]] = []
    for r in results:
        for f in r.flags:
            cid = len(comment_data)
            comment_data.append((cid, f"[{f.type}] {f.issue}"))
            anchors.append((f.text, cid))

    if not comment_data:
        return 0

    xml_bytes     = _make_comment_xml(comment_data)
    comments_part = Part(PackURI(COMMENTS_URI), COMMENTS_CT, xml_bytes)
    doc.part.relate_to(comments_part, COMMENTS_RT)

    paragraphs = [p._element for p in doc.paragraphs if p.text.strip()]
    anchored   = 0

    for search, cid in anchors:
        # ── Pass 1: exact (or progressively shortened) span match ─────────────
        found = False
        for para_el in paragraphs:
            if _anchor_comment(para_el, search, cid):
                found = True
                anchored += 1
                break

        if found:
            continue

        # ── Pass 2: paragraph-level fallback — only if ALL significant words
        #    from the search string appear in the paragraph text ───────────────
        sig = _sig_words(search)
        if not sig:
            continue

        for para_el in paragraphs:
            runs = _para_runs(para_el)
            if not runs:
                continue
            para_text = _norm("".join(_run_text(r) for r in runs)).lower()
            if all(w in para_text for w in sig):
                runs[0].addprevious(_comment_range_start(cid))
                runs[-1].addnext(_comment_ref_run(cid))
                runs[-1].addnext(_comment_range_end(cid))
                anchored += 1
                break

    doc.save(output_path)
    return anchored
