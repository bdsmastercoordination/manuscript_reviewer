import os
import re
import ctypes
import threading
import tkinter as tk
from tkinter import filedialog, ttk, font as tkfont

# Must be called before the Tk window is created
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageTk

from core.reviewer import review_document, ChunkResult, REVIEWER_MODEL
from core.docx_comments import annotate
from core.prompts import load_error_types, load_style_config, save_style_config
from core.character_analyzer import analyze_characters, load_archetypes, CharacterSheet
from core.emotion_analyzer import analyze_emotions, load_emotions
from core.personality_analyzer import analyze_personality, load_personality_traits
from core.genre_analyzer import analyze_genres, GenreResult, load_genres
from core.synopsis_analyzer import analyze_synopsis, SynopsisResult, PlotBeat
from core.opening_analyzer import analyze_opening, OpeningResult, CHECKLIST_ITEMS, CONFIDENCE_EMOJI
from core.arc_analyzer import analyze_arcs, ArcsResult, ArcResult, load_structural_arcs
from core.cache import (
    load_copyedit_cache, save_copyedit_cache,
    load_arch_cache, save_arch_cache,
    load_emotion_cache, save_emotion_cache,
    load_personality_cache, save_personality_cache,
    load_genre_cache, save_genre_cache,
    load_synopsis_cache, save_synopsis_cache,
    load_opening_cache, save_opening_cache,
    load_arcs_cache, save_arcs_cache,
)
from config.archetype_icons import ICON_PATH as ARCH_ICON_PATH

EMOTION_ICONS_DIR = Path(__file__).parent / "assets" / "emotions"
_EMOTION_CUSTOM_ICON = EMOTION_ICONS_DIR / "custom.png"


def _emotion_icon_path(eid: str) -> Path:
    """Return the icon path for an emotion ID, falling back to custom.jpg."""
    p = EMOTION_ICONS_DIR / f"{eid}.png"
    if p.exists():
        return p
    return _EMOTION_CUSTOM_ICON


PERSONALITY_ICONS_DIR = Path(__file__).parent / "assets" / "personalities" / "individual"
_PERSONALITY_CUSTOM_ICON = PERSONALITY_ICONS_DIR / "custom.png"


def _personality_icon_path(tid: str, score: float = 1.0) -> Path:
    pole = "high" if score >= 0 else "low"
    p = PERSONALITY_ICONS_DIR / f"{tid}_{pole}.png"
    if p.exists():
        return p
    p2 = PERSONALITY_ICONS_DIR / f"{tid}.png"
    if p2.exists():
        return p2
    return _PERSONALITY_CUSTOM_ICON


# ── palette ───────────────────────────────────────────────────────────────────
# BG matches the logo's own background (#fdf9f6) — seamless on landing and app
BG          = "#fdf9f6"   # logo cream — outer surfaces and toolbars
BG_RESULTS  = "#ffffff"   # white for the scrollable output fields
BG_SETTINGS = "#f0e3ca"   # warm sand for settings panels
ACCENT      = "#c9a96e"   # warm gold (the Analyze button colour)
ACCENT_HOV  = "#b8954e"   # deeper gold on hover
BTN_SEC     = "#ede0c8"   # sand secondary button
BTN_SEC_H   = "#e2d0ae"   # secondary hover
BTN_BORDER  = "#c9a96e"   # gold outline — ties to accent
TEXT        = "#111111"   # near-black
TEXT_MUTED  = "#6b7280"   # neutral gray
TEXT_TYPE   = "#1e40af"   # blue for category / archetype labels
BORDER      = "#d4ba90"   # gold-tinted divider

# ── typography ────────────────────────────────────────────────────────────────
FONT_UI    = ("Georgia", 11)
FONT_TITLE = ("Georgia", 14, "bold")
FONT_TYPE  = ("Georgia", 11, "bold")
FONT_BODY  = ("Georgia", 11)
FONT_QUOTE = ("Georgia", 11, "italic")
FONT_SMALL = ("Georgia", 10)
FONT_LABEL = ("Georgia", 10, "bold")
FONT_COG   = ("Segoe UI", 12)   # gear symbol not in Georgia

ICON_PATH = Path(__file__).parent / "assets" / "logo.png"

MODELS = {
    "Haiku (fast)":  "claude-haiku-4-5-20251001",
    "Sonnet (best)": "claude-sonnet-4-6",
}

CHAR_COLORS = ["#c9a96e", "#4a7ebf", "#7ca87c", "#b86b6b", "#9b7ec8", "#c8a07c"]

PLUTCHIK_ORDER = ["joy", "love", "fear", "surprise", "longing",
                  "sadness", "disgust", "anger", "pride", "shame"]

STYLE_LABELS = {
    "quotation_marks": "Quotation marks",
    "ellipsis":        "Ellipsis style",
    "em_dash_spacing": "Em dash spacing",
    "oxford_comma":    "Oxford comma",
    "language":        "Language",
}


def _fmt_cache_dt(iso: str) -> str:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%d %b %Y, %H:%M")
    except Exception:
        return iso


def _softmax_sizes(scores: list[float], lo: int = 52, hi: int = 96) -> list[int]:
    """Map scores to pixel sizes via softmax: highest prob → hi, lowest → lo."""
    import math
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    p_min, p_max = min(probs), max(probs)
    if p_max == p_min:
        mid = (lo + hi) // 2
        return [mid] * len(scores)
    return [round(lo + (p - p_min) / (p_max - p_min) * (hi - lo)) for p in probs]


def _flat_btn(parent, text, command, primary=False, **kw):
    bg  = ACCENT    if primary else BTN_SEC
    fg  = TEXT      if primary else TEXT
    hov = ACCENT_HOV if primary else BTN_SEC_H
    hl  = ACCENT    if primary else BTN_BORDER
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=FONT_UI,
        relief=tk.FLAT, bd=0, padx=12, pady=5,
        cursor="hand2", activebackground=hov, activeforeground=fg,
        highlightthickness=1, highlightbackground=hl,
        **kw
    )
    btn.bind("<Enter>", lambda _: btn.config(bg=hov))
    btn.bind("<Leave>", lambda _: btn.config(bg=bg))
    return btn


def _make_group_toggle(parent, var, label_a, val_a, label_b, val_b, on_change=None):
    """
    Segmented pill toggle with two options.  Active side gets ACCENT background.
    Returns the outer tk.Frame.
    """
    frame = tk.Frame(
        parent, bg=BG_SETTINGS,
        highlightthickness=1, highlightbackground=BORDER,
    )

    def _set(val):
        var.set(val)
        _refresh()
        if on_change:
            on_change()

    lbl_a = tk.Label(frame, text=label_a, font=FONT_SMALL,
                     padx=14, pady=5, cursor="hand2")
    lbl_a.pack(side=tk.LEFT)
    lbl_a.bind("<Button-1>", lambda _: _set(val_a))

    tk.Frame(frame, width=1, bg=BORDER).pack(side=tk.LEFT, fill=tk.Y)

    lbl_b = tk.Label(frame, text=label_b, font=FONT_SMALL,
                     padx=14, pady=5, cursor="hand2")
    lbl_b.pack(side=tk.LEFT)
    lbl_b.bind("<Button-1>", lambda _: _set(val_b))

    def _refresh():
        mode = var.get()
        lbl_a.config(
            bg=ACCENT     if mode == val_a else BG_SETTINGS,
            fg="#ffffff"  if mode == val_a else TEXT_MUTED,
        )
        lbl_b.config(
            bg=ACCENT     if mode == val_b else BG_SETTINGS,
            fg="#ffffff"  if mode == val_b else TEXT_MUTED,
        )

    _refresh()
    return frame


def _draw_arch_bars(canvas, archetypes, width):
    """Draw proportional horizontal bars onto a tk.Canvas."""
    canvas.delete("all")
    if not archetypes or width < 20:
        return
    max_score = max(s for _, _, s in archetypes) or 1
    row_h, bar_h = 26, 11
    label_w = min(int(width * 0.38), 238)
    score_w = 48
    bar_area = min(max(width - label_w - score_w - 12, 10), 280)

    font_obj = tkfont.Font(font=FONT_SMALL)
    for i, (arch_id, name, score) in enumerate(archetypes):
        y_mid  = 4 + i * row_h + row_h // 2
        bar_y1 = y_mid - bar_h // 2
        bar_y2 = bar_y1 + bar_h

        avail = label_w - 6
        label = name
        while label and font_obj.measure(label) > avail:
            label = label[:-1]
        if label != name:
            while label and font_obj.measure(label + "…") > avail:
                label = label[:-1]
            label = label + "…"
        canvas.create_text(0, y_mid, text=label, font=FONT_SMALL,
                           fill=TEXT, anchor="w")
        canvas.create_rectangle(label_w, bar_y1, label_w + bar_area, bar_y2,
                                fill=BTN_SEC, outline="")
        fill_w = int(bar_area * score / max_score)
        if fill_w > 0:
            canvas.create_rectangle(label_w, bar_y1, label_w + fill_w, bar_y2,
                                    fill=ACCENT, outline="")
        canvas.create_text(label_w + bar_area + 6, y_mid,
                           text=f"{score:.2f}", font=FONT_SMALL,
                           fill=TEXT_MUTED, anchor="w")


def _build_arch_fig(sheets, total_chunks=None):
    """
    Archetype time-series figure.
    Raw net scores (pos − neg), no smoothing, dynamic y-axis.
    Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None, []

    LINE_COLORS = ["#c9a96e", "#7c5c1e", "#d4956a"]

    all_cidxs = [s["chunk_idx"] for sh in chars for s in sh.signals_ordered]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n    = len(chars)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.4, n * 2.2)),
                             facecolor=BG, squeeze=False)
    axes  = axes[:, 0]
    marked = []

    for ax, sheet in zip(axes, chars):
        pos_sigs = sheet.signals_ordered
        ax_yvals = []
        if not pos_sigs:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)
            continue

        x_dense     = np.linspace(0, 100, total_chunks)
        chunk_width = 100.0 / max(total_chunks - 1, 1)
        margin      = chunk_width * 0.25
        n_fine      = total_chunks * 4
        x_fine      = np.linspace(0, 100, n_fine)
        top3        = sheet.top_archetypes[:3]

        ax.set_facecolor("#faf5ec")
        ax.axhline(0, color=BORDER, linewidth=0.8, zorder=1)
        any_line = False

        neg_sigs = sheet.neg_signals_ordered

        for ci_color, (aid, aname, _) in enumerate(top3):
            # Positive signals (score ≥ 0.5)
            chunk_to_pos: dict[int, list] = {}
            for s in pos_sigs:
                if s["scores"].get(aid, 0.0) < 0.5:
                    continue
                chunk_to_pos.setdefault(s["chunk_idx"], []).append(s)

            # Negative signals (violation score ≥ 4, stored as positive; negate for plotting)
            chunk_to_neg: dict[int, list] = {}
            for s in neg_sigs:
                if s["scores"].get(aid, 0.0) < 4.0:
                    continue
                chunk_to_neg.setdefault(s["chunk_idx"], []).append(s)

            if not chunk_to_pos and not chunk_to_neg:
                continue

            # Build spread_data: merge pos + neg per chunk, ordered by narrative offset
            spread_data: list[tuple] = []
            for ci_s in sorted(set(list(chunk_to_pos) + list(chunk_to_neg))):
                combined = sorted(
                    [(sig, False) for sig in chunk_to_pos.get(ci_s, [])] +
                    [(sig, True)  for sig in chunk_to_neg.get(ci_s, [])],
                    key=lambda t: t[0].get("offset", 50),
                )
                cx = float(x_dense[ci_s])
                lo = max(0.0,   cx - chunk_width * 0.5 + margin)
                hi = min(100.0, cx + chunk_width * 0.5 - margin)
                for sig, is_neg in combined:
                    off = sig.get("offset", 50) / 100.0
                    xi  = lo + off * (hi - lo)
                    spread_data.append((xi, ci_s, sig, is_neg))

            xs_arr = np.array([d[0] for d in spread_data])
            ys_arr = np.array([
                -d[2]["scores"].get(aid, 0.0) if d[3] else d[2]["scores"].get(aid, 0.0)
                for d in spread_data
            ])

            # 0.4 self + 0.3 prev + 0.3 next; redistribute missing neighbour weight to self
            n_sig   = len(ys_arr)
            _self_w = lambda i: 1.0 if n_sig == 1 else (0.7 if i in (0, n_sig - 1) else 0.4)
            ys_arr  = np.array([
                _self_w(i) * ys_arr[i]
                + (0.3 * ys_arr[i - 1] if i > 0         else 0.0)
                + (0.3 * ys_arr[i + 1] if i < n_sig - 1 else 0.0)
                for i in range(n_sig)
            ])

            if np.abs(ys_arr).max() < 0.1:
                continue

            net_fine = np.interp(x_fine, xs_arr, ys_arr,
                                 left=ys_arr[0], right=ys_arr[-1])

            all_cis   = [d[1] for d in spread_data]
            x_start   = max(0.0,   float(x_dense[min(all_cis)]) - chunk_width * 0.5)
            x_end     = min(100.0, float(x_dense[max(all_cis)]) + chunk_width * 0.5)
            line_mask = (x_fine >= x_start) & (x_fine <= x_end)

            col = LINE_COLORS[ci_color % len(LINE_COLORS)]
            ax.plot(x_fine[line_mask], net_fine[line_mask], color=col, linewidth=1.8,
                    label=aname, alpha=0.92, solid_capstyle="round", zorder=3)
            ax_yvals.extend(net_fine[line_mask].tolist())
            any_line = True

            for xi, ci_s, sig, is_neg in spread_data:
                yi = float(np.interp(xi, x_fine, net_fine))
                ax_yvals.append(yi)
                ax.plot(xi, yi, "o", color=col, markersize=4.5,
                        markeredgewidth=0.8, markeredgecolor="white", alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=yi, aid=aid, aname=aname,
                                   chunk_idx=ci_s, sheet=sheet, signal=sig, is_neg=is_neg))

        if not any_line:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if ax_yvals:
            ax.set_ylim(min(ax_yvals) - 0.5, max(ax_yvals) + 0.5)
        ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
        ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        ax.set_ylabel("Archetype Fit", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BORDER)
        if any_line:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8,
                      facecolor=BG, edgecolor=BORDER, handlelength=1.4)

    fig.tight_layout(pad=0.9, h_pad=1.4)
    return fig, marked


def _build_emotion_fig(sheets, total_chunks=None):
    """
    Emotion time-series figure.
    Weighted 3-point smoothing with emotion-valence sentiment, y-axis 0–10.
    Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None, []

    LINE_COLORS = ["#c9a96e", "#7c5c1e", "#d4956a"]
    EMOTION_POS = {"joy", "love", "pride"}
    EMOTION_NEG = {"sadness", "anger", "fear", "shame", "disgust"}

    all_cidxs = [s["chunk_idx"] for sh in chars for s in sh.signals_ordered]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n    = len(chars)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.4, n * 2.2)),
                             facecolor=BG, squeeze=False)
    axes  = axes[:, 0]
    marked = []

    def _signal_sentiment(s: dict) -> float:
        pos = [sc for eid, sc in s["scores"].items() if eid in EMOTION_POS]
        neg = [sc for eid, sc in s["scores"].items() if eid in EMOTION_NEG]
        return (sum(pos) / len(pos) if pos else 0.0) - (sum(neg) / len(neg) if neg else 0.0)

    for ax, sheet in zip(axes, chars):
        pos_sigs = sheet.signals_ordered
        ax_yvals = []
        if not pos_sigs:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)
            continue

        x_dense     = np.linspace(0, 100, total_chunks)
        chunk_width = 100.0 / max(total_chunks - 1, 1)
        margin      = chunk_width * 0.25
        n_fine      = total_chunks * 4
        x_fine      = np.linspace(0, 100, n_fine)
        top3        = sheet.top_archetypes[:3]

        # Chunk-level emotion sentiment
        _by_chunk: dict[int, list] = {}
        for s in pos_sigs:
            _by_chunk.setdefault(s["chunk_idx"], []).append(s)
        chunk_sentiment: dict[int, float] = {
            ci_s: float(np.mean([_signal_sentiment(s) for s in sigs]))
            for ci_s, sigs in _by_chunk.items()
        }

        ax.set_facecolor("#faf5ec")
        any_line = False

        for ci_color, (aid, aname, _) in enumerate(top3):
            chunk_to_sigs: dict[int, list] = {}
            for s in pos_sigs:
                if s["scores"].get(aid, 0.0) < 0.5:
                    continue
                chunk_to_sigs.setdefault(s["chunk_idx"], []).append(s)
            if not chunk_to_sigs:
                continue

            spread_data: list[tuple] = []
            for ci_s in sorted(chunk_to_sigs):
                c_sigs = sorted(chunk_to_sigs[ci_s], key=lambda s: s.get("offset", 50))
                cx = float(x_dense[ci_s])
                lo = max(0.0,   cx - chunk_width * 0.5 + margin)
                hi = min(100.0, cx + chunk_width * 0.5 - margin)
                for sig in c_sigs:
                    off = sig.get("offset", 50) / 100.0
                    xi  = lo + off * (hi - lo)
                    spread_data.append((xi, ci_s, sig))

            xs_arr = np.array([d[0] for d in spread_data])
            ys_arr = np.array([d[2]["scores"].get(aid, 0.0) for d in spread_data])

            # Weighted smoothing: 60% self, 10% prev, 10% next, 20% chunk sentiment
            n_sig     = len(ys_arr)
            ys_smooth = np.array([
                0.6 * ys_arr[i]
                + 0.1 * ys_arr[max(0, i - 1)]
                + 0.1 * ys_arr[min(n_sig - 1, i + 1)]
                + 0.2 * chunk_sentiment.get(spread_data[i][1], 0.0)
                for i in range(n_sig)
            ])

            if np.abs(ys_smooth).max() < 0.1:
                continue

            net_fine = np.interp(x_fine, xs_arr, ys_smooth,
                                 left=ys_smooth[0], right=ys_smooth[-1])

            first_ci  = spread_data[0][1]
            last_ci   = spread_data[-1][1]
            x_start   = max(0.0,   float(x_dense[first_ci]) - chunk_width * 0.5)
            x_end     = min(100.0, float(x_dense[last_ci])  + chunk_width * 0.5)
            line_mask = (x_fine >= x_start) & (x_fine <= x_end)

            col = LINE_COLORS[ci_color % len(LINE_COLORS)]
            ax.plot(x_fine[line_mask], net_fine[line_mask], color=col, linewidth=1.8,
                    label=aname, alpha=0.92, solid_capstyle="round", zorder=3)
            ax_yvals.extend(net_fine[line_mask].tolist())
            any_line = True

            for xi, ci_s, sig in spread_data:
                yi = float(np.interp(xi, x_fine, net_fine))
                ax_yvals.append(yi)
                ax.plot(xi, yi, "o", color=col, markersize=4.5,
                        markeredgewidth=0.8, markeredgecolor="white", alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=yi, aid=aid, aname=aname,
                                   chunk_idx=ci_s, sheet=sheet, signal=sig))

        if not any_line:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if ax_yvals:
            ax.set_ylim(min(ax_yvals) - 0.5, max(ax_yvals) + 0.5)
        ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
        ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        ax.set_ylabel("Emotion", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BORDER)

        if any_line:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8,
                      facecolor=BG, edgecolor=BORDER, handlelength=1.4)

    fig.tight_layout(pad=0.9, h_pad=1.4)
    return fig, marked


def _render_archetype_plot(sheets):
    """Render archetype time-series to a PIL Image (for PDF embedding)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None
    fig, _ = _build_arch_fig(sheets)
    if fig is None:
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#fdf9f6")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _make_arch_plot_widget(parent, sheets, total_chunks=None):
    """
    Embed an interactive archetype plot inside `parent`.
    Dots mark the smoothed peak and valley of each archetype line;
    clicking near one opens a popup with the underlying signal quote.
    Returns the containing Frame, or None if matplotlib is unavailable.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_arch_fig(sheets, total_chunks=total_chunks)
    if fig is None:
        return None

    frame = tk.Frame(parent, bg=BG_RESULTS)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1)
        yr   = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            tk_widget.config(cursor="hand2")
        else:
            tk_widget.config(cursor="")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sheet     = point["sheet"]
        aid       = point["aid"]
        ci        = point["chunk_idx"]
        aname     = point["aname"]

        sig    = point["signal"]
        is_neg = point.get("is_neg", False)
        score  = sig["scores"].get(aid, 0.0)

        popup = tk.Toplevel(parent)
        popup.title(aname)
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()

        tk.Label(popup, text=f"{sheet.name}  ·  {aname}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))

        body = tk.Frame(popup, bg=BG)
        body.pack(fill=tk.X, padx=24, pady=(0, 6))

        row = tk.Frame(body, bg=BG)
        row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row, text=sig.get("signal", ""), font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=400, justify=tk.LEFT, anchor="nw").pack(
                     fill=tk.X, expand=True)

        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))

        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root = parent.winfo_toplevel()
        x = root.winfo_rootx() + (root.winfo_width()  - pw) // 2
        y = root.winfo_rooty() + (root.winfo_height() - ph) // 2
        popup.geometry(f"+{x}+{y}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)   # hand ownership to the canvas; avoid double-close issues
    return frame


def _render_emotion_plot(sheets):
    """Render emotion time-series to a PIL Image (for PDF embedding)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None
    fig, _ = _build_emotion_fig(sheets)
    if fig is None:
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _make_emotion_plot_widget(parent, sheets, total_chunks=None):
    """Interactive emotion line-plot widget; shares interaction code with arch variant."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_emotion_fig(sheets, total_chunks=total_chunks)
    if fig is None:
        return None

    frame     = tk.Frame(parent, bg=BG_RESULTS)
    canvas    = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1); yr = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        tk_widget.config(cursor="hand2" if best and best_dist < 0.01 else "")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sig   = point["signal"]
        aid   = point["aid"]
        score = sig["scores"].get(aid, 0.0)
        popup = tk.Toplevel(parent)
        popup.title(point["aname"])
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()
        tk.Label(popup, text=f"{point['sheet'].name}  ·  {point['aname']}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))
        body = tk.Frame(popup, bg=BG)
        body.pack(fill=tk.X, padx=24, pady=(0, 6))
        row = tk.Frame(body, bg=BG)
        row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row, text=sig.get("signal", ""), font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=400, justify=tk.LEFT, anchor="nw").pack(
                     fill=tk.X, expand=True)
        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))
        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root   = parent.winfo_toplevel()
        popup.geometry(f"+{root.winfo_rootx() + (root.winfo_width()  - pw) // 2}"
                       f"+{root.winfo_rooty() + (root.winfo_height() - ph) // 2}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)
    return frame


def _make_emotion_radar_widget(parent, sheet, extra_emotions=None) -> tk.Frame | None:
    """
    Render a polar spider-web (radar) chart for one character's emotion profile.
    Scores are reconstructed from signals_ordered so all emotion spokes are shown.
    Returns a tk.Frame containing the embedded chart, or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    emotions = load_emotions()
    if extra_emotions:
        known_ids = {e["id"] for e in emotions}
        emotions = emotions + [e for e in extra_emotions if e["id"] not in known_ids]
    if not emotions:
        return None

    # Clockwise from top; opposite spokes (diff=5) match Plutchik poles:
    # Joy↔Sadness, Love(Trust)↔Disgust, Fear↔Anger
    PLUTCHIK_ORDER = ["joy", "love", "fear", "surprise", "longing",
                      "sadness", "disgust", "anger", "pride", "shame"]
    by_id   = {e["id"]: e for e in emotions}
    ordered = [by_id[eid] for eid in PLUTCHIK_ORDER if eid in by_id]
    # Append any emotions not covered by the fixed order (including custom ones)
    ordered += [e for e in emotions if e["id"] not in PLUTCHIK_ORDER]

    all_ids = [e["id"] for e in ordered]
    labels  = [e["name"] for e in ordered]
    N = len(ordered)

    # Reconstruct full avg score vector from signals_ordered
    totals = {eid: 0.0 for eid in all_ids}
    T = sheet.total_signals or 1
    for sig in sheet.signals_ordered:
        for eid, score in sig.get("scores", {}).items():
            if eid in totals:
                totals[eid] += score
    raw    = [totals[eid] / T for eid in all_ids]
    floor  = max(raw) * 0.06 if max(raw) > 0 else 0
    values = [max(v, floor) for v in raw]

    if max(values) < 0.05:
        return None

    # Polar layout
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    v_plot = values + [values[0]]
    a_plot = angles + [angles[0]]

    FIG_BG = "#fdf9f6"
    AX_BG  = "#faf5ec"
    GOLD   = "#c9a96e"
    BORDER = "#d4ba90"

    fig, ax = plt.subplots(
        figsize=(3.4, 3.4),
        subplot_kw=dict(polar=True),
        facecolor=FIG_BG,
    )
    ax.set_facecolor(AX_BG)

    # Concentric grid rings
    max_v = max(values)
    ticks = np.linspace(0, max_v, 4)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_ylim(0, max_v * 1.15)
    ax.yaxis.grid(color=BORDER, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.grid(color=BORDER, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(BORDER)

    # Filled area
    ax.fill(a_plot, v_plot, color=GOLD, alpha=0.22)
    ax.plot(a_plot, v_plot, color=GOLD, linewidth=1.6)

    # Spoke dots
    ax.scatter(angles, values, color=GOLD, s=18, zorder=4, edgecolors="white", linewidths=0.8)

    # Labels — push them slightly outside the ring
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.5, color="#111111")

    fig.tight_layout(pad=0.3)

    frame = tk.Frame(parent, bg=BG)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    return frame


def _build_personality_fig(sheets, total_chunks=None):
    """
    Big Five personality trait time-series figure.
    Plots every individual signal as a dot (clickable) and interpolates a line.
    Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None, []

    LINE_COLORS = ["#c9a96e", "#7c5c1e", "#d4956a"]

    all_cidxs = [s["chunk_idx"] for sh in chars for s in sh.signals_ordered]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n = len(chars)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.4, n * 2.2)),
                             facecolor=BG, squeeze=False)
    axes  = axes[:, 0]
    marked = []

    for ax, sheet in zip(axes, chars):
        pos_sigs = sheet.signals_ordered
        ax_yvals = []
        if not pos_sigs:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)
            continue

        x_dense     = np.linspace(0, 100, total_chunks)
        chunk_width = 100.0 / max(total_chunks - 1, 1)
        margin      = chunk_width * 0.25
        n_fine      = total_chunks * 4
        x_fine      = np.linspace(0, 100, n_fine)
        top3        = sheet.top_archetypes[:3]

        ax.set_facecolor("#faf5ec")
        any_line = False

        for ci_color, (tid, tname, _) in enumerate(top3):
            chunk_to_sigs: dict[int, list] = {}
            for s in pos_sigs:
                if abs(s["scores"].get(tid, 0.0)) < 0.5:
                    continue
                chunk_to_sigs.setdefault(s["chunk_idx"], []).append(s)
            if not chunk_to_sigs:
                continue

            spread_data: list[tuple] = []
            for ci_s in sorted(chunk_to_sigs):
                c_sigs = sorted(chunk_to_sigs[ci_s], key=lambda s: s.get("offset", 50))
                cx = float(x_dense[ci_s])
                lo = max(0.0,   cx - chunk_width * 0.5 + margin)
                hi = min(100.0, cx + chunk_width * 0.5 - margin)
                for sig in c_sigs:
                    off = sig.get("offset", 50) / 100.0
                    xi  = lo + off * (hi - lo)
                    spread_data.append((xi, ci_s, sig))

            if not spread_data:
                continue

            xs_arr = np.array([d[0] for d in spread_data])
            # use signed scores for y-axis (bipolar)
            ys_arr = np.array([d[2]["scores"].get(tid, 0.0) for d in spread_data])

            if np.max(np.abs(ys_arr)) < 0.1:
                continue

            net_fine = np.interp(x_fine, xs_arr, ys_arr,
                                 left=ys_arr[0], right=ys_arr[-1])

            first_ci = spread_data[0][1]
            last_ci  = spread_data[-1][1]
            x_start  = max(0.0,   float(x_dense[first_ci]) - chunk_width * 0.5)
            x_end    = min(100.0, float(x_dense[last_ci])  + chunk_width * 0.5)
            line_mask = (x_fine >= x_start) & (x_fine <= x_end)

            col = LINE_COLORS[ci_color % len(LINE_COLORS)]
            ax.plot(x_fine[line_mask], net_fine[line_mask], color=col, linewidth=1.8,
                    label=tname, alpha=0.92, solid_capstyle="round", zorder=3)
            ax_yvals.extend(net_fine[line_mask].tolist())
            any_line = True

            for xi, ci_s, sig in spread_data:
                yi = float(np.interp(xi, x_fine, net_fine))
                ax_yvals.append(yi)
                ax.plot(xi, yi, "o", color=col, markersize=4.5,
                        markeredgewidth=0.8, markeredgecolor="white", alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=yi, aid=tid, aname=tname,
                                   chunk_idx=ci_s, sheet=sheet, signal=sig))

        if not any_line:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)

        # zero baseline for bipolar scale
        ax.axhline(0, color=BORDER, linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if ax_yvals:
            ylo = min(ax_yvals + [-0.2])
            yhi = max(ax_yvals + [0.2])
            pad = (yhi - ylo) * 0.15 or 0.3
            ax.set_ylim(ylo - pad, yhi + pad)
        ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
        ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        ax.set_ylabel("← low  ·  high →", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BORDER)

        if any_line:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8,
                      facecolor=BG, edgecolor=BORDER, handlelength=1.4)
        ax.set_title(sheet.name, fontsize=9, color=TEXT, loc="left", pad=4)

    fig.tight_layout(pad=0.9, h_pad=1.4)
    return fig, marked


def _render_personality_plot(sheets):
    """Render personality trait time-series to a PIL Image (for PDF embedding)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None
    fig, _ = _build_personality_fig(sheets)
    if fig is None:
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _make_personality_plot_widget(parent, sheets, total_chunks=None):
    """Interactive personality trait line-plot widget; clickable signal dots."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_personality_fig(sheets, total_chunks=total_chunks)
    if fig is None:
        return None

    frame     = tk.Frame(parent, bg=BG_RESULTS)
    canvas    = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1); yr = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        tk_widget.config(cursor="hand2" if best and best_dist < 0.01 else "")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sig   = point["signal"]
        aid   = point["aid"]
        popup = tk.Toplevel(parent)
        popup.title(point["aname"])
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()
        tk.Label(popup, text=f"{point['sheet'].name}  ·  {point['aname']}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))
        body = tk.Frame(popup, bg=BG)
        body.pack(fill=tk.X, padx=24, pady=(0, 6))
        tk.Label(body, text=sig.get("signal", ""), font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=400, justify=tk.LEFT, anchor="nw").pack(fill=tk.X)
        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))
        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root   = parent.winfo_toplevel()
        popup.geometry(f"+{root.winfo_rootx() + (root.winfo_width()  - pw) // 2}"
                       f"+{root.winfo_rooty() + (root.winfo_height() - ph) // 2}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)
    return frame


def _make_personality_radar_widget(parent, sheet, extra_traits=None) -> tk.Frame | None:
    """
    Render a polar radar chart for one character's Big Five profile.
    Returns a tk.Frame or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    traits = load_personality_traits()
    if extra_traits:
        known_ids = {t["id"] for t in traits}
        traits = traits + [t for t in extra_traits if t["id"] not in known_ids]
    if not traits:
        return None

    all_ids  = [t["id"] for t in traits]
    # pole name per trait — shown when the character leans to that pole
    high_names = [t.get("high_name", t["name"]) for t in traits]
    low_names  = [t.get("low_name",  t["name"]) for t in traits]
    N = len(traits)

    totals = {tid: 0.0 for tid in all_ids}
    T = sheet.total_signals or 1
    for sig in sheet.signals_ordered:
        for tid, score in sig.get("scores", {}).items():
            if tid in totals:
                totals[tid] += score
    avg    = [totals[tid] / T for tid in all_ids]
    # radar uses |avg| for spoke length; sign selects the pole label
    abs_avg = [abs(v) for v in avg]
    labels  = [
        high_names[i] if avg[i] >= 0 else low_names[i]
        for i in range(N)
    ]
    floor  = max(abs_avg) * 0.06 if max(abs_avg) > 0 else 0
    values = [max(v, floor) for v in abs_avg]

    if max(values) < 0.05:
        return None

    angles = [n / float(N) * 2 * 3.14159265 for n in range(N)]
    v_plot = values + [values[0]]
    a_plot = angles + [angles[0]]

    FIG_BG = "#fdf9f6"
    AX_BG  = "#faf5ec"
    GOLD   = "#c9a96e"
    BORD   = "#d4ba90"

    fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw=dict(polar=True), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    max_v = max(values)
    import numpy as np
    ticks = np.linspace(0, max_v, 4)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_ylim(0, max_v * 1.15)
    ax.yaxis.grid(color=BORD, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.grid(color=BORD, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(BORD)

    ax.fill(a_plot, v_plot, color=GOLD, alpha=0.22)
    ax.plot(a_plot, v_plot, color=GOLD, linewidth=1.6)
    ax.scatter(angles, values, color=GOLD, s=18, zorder=4, edgecolors="white", linewidths=0.8)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.5, color="#111111")

    fig.tight_layout(pad=0.3)

    frame  = tk.Frame(parent, bg=BG)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    return frame


def _make_combined_emotion_radar_widget(parent, sheets, extra_emotions=None) -> "tk.Frame | None":
    """
    Render a radar chart with all characters overlaid (one polygon per character).
    Returns a tk.Frame or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None

    emotions = load_emotions()
    if extra_emotions:
        known_ids = {e["id"] for e in emotions}
        emotions = emotions + [e for e in extra_emotions if e["id"] not in known_ids]
    if not emotions:
        return None

    by_id   = {e["id"]: e for e in emotions}
    ordered = [by_id[eid] for eid in PLUTCHIK_ORDER if eid in by_id]
    ordered += [e for e in emotions if e["id"] not in PLUTCHIK_ORDER]

    all_ids = [e["id"] for e in ordered]
    labels  = [e["name"] for e in ordered]
    N = len(ordered)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    FIG_BG = "#fdf9f6"
    AX_BG  = "#faf5ec"
    BORD   = "#d4ba90"

    fig, ax = plt.subplots(
        figsize=(4.2, 4.2),
        subplot_kw=dict(polar=True),
        facecolor=FIG_BG,
    )
    ax.set_facecolor(AX_BG)

    all_values = []
    char_data  = []
    for sheet in chars:
        totals = {eid: 0.0 for eid in all_ids}
        T = sheet.total_signals or 1
        for sig in sheet.signals_ordered:
            for eid, score in sig.get("scores", {}).items():
                if eid in totals:
                    totals[eid] += score
        raw    = [totals[eid] / T for eid in all_ids]
        floor  = max(raw) * 0.06 if max(raw) > 0 else 0
        values = [max(v, floor) for v in raw]
        all_values.extend(values)
        char_data.append((sheet.name, values))

    if not all_values or max(all_values) < 0.05:
        plt.close(fig)
        return None

    max_v = max(all_values)
    ticks = np.linspace(0, max_v, 4)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_ylim(0, max_v * 1.18)
    ax.yaxis.grid(color=BORD, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.grid(color=BORD, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(BORD)

    for ci, (cname, values) in enumerate(char_data):
        col    = CHAR_COLORS[ci % len(CHAR_COLORS)]
        v_plot = values + [values[0]]
        a_plot = angles + [angles[0]]
        ax.fill(a_plot, v_plot, color=col, alpha=0.18)
        ax.plot(a_plot, v_plot, color=col, linewidth=1.6, label=cname)
        ax.scatter(angles, values, color=col, s=14, zorder=4, edgecolors="white", linewidths=0.6)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.5, color="#111111")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.32, 1.15),
              framealpha=0.85, facecolor=FIG_BG, edgecolor=BORD)

    fig.tight_layout(pad=0.3)

    frame  = tk.Frame(parent, bg=BG)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    return frame


def _build_emotion_by_group_fig(sheets, total_chunks=None):
    """
    Emotion by-group time-series figure.
    One subplot per emotion (top_archetypes[:3] across all characters).
    One line per character, color-coded.
    Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None, []

    EMOTION_POS = {"joy", "love", "pride"}
    EMOTION_NEG = {"sadness", "anger", "fear", "shame", "disgust"}

    def _signal_sentiment(s: dict) -> float:
        pos = [sc for eid, sc in s["scores"].items() if eid in EMOTION_POS]
        neg = [sc for eid, sc in s["scores"].items() if eid in EMOTION_NEG]
        return (sum(pos) / len(pos) if pos else 0.0) - (sum(neg) / len(neg) if neg else 0.0)

    # Collect unique emotion IDs from top_archetypes[:3] across all chars
    emotion_ids_ordered = []
    seen = set()
    for sheet in chars:
        for aid, aname, _ in sheet.top_archetypes[:3]:
            if aid not in seen:
                seen.add(aid)
                emotion_ids_ordered.append((aid, aname))

    if not emotion_ids_ordered:
        return None, []

    all_cidxs = [s["chunk_idx"] for sh in chars for s in sh.signals_ordered]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n_emot = len(emotion_ids_ordered)
    fig, axes = plt.subplots(n_emot, 1, figsize=(9, max(2.4, n_emot * 2.2)),
                              facecolor=BG, squeeze=False)
    axes  = axes[:, 0]
    marked = []

    x_dense     = np.linspace(0, 100, total_chunks)
    chunk_width = 100.0 / max(total_chunks - 1, 1)
    margin      = chunk_width * 0.25
    n_fine      = total_chunks * 4
    x_fine      = np.linspace(0, 100, n_fine)

    for ax, (aid, aname) in zip(axes, emotion_ids_ordered):
        ax.set_facecolor("#faf5ec")
        ax.set_title(aname, fontsize=9, color=TEXT, loc="left", pad=4)
        ax_yvals = []
        any_line = False

        for ci_char, sheet in enumerate(chars):
            col    = CHAR_COLORS[ci_char % len(CHAR_COLORS)]
            pos_sigs = sheet.signals_ordered

            # Chunk sentiment for smoothing
            _by_chunk: dict[int, list] = {}
            for s in pos_sigs:
                _by_chunk.setdefault(s["chunk_idx"], []).append(s)
            chunk_sentiment: dict[int, float] = {
                ci_s: float(np.mean([_signal_sentiment(s) for s in sigs]))
                for ci_s, sigs in _by_chunk.items()
            }

            chunk_to_sigs: dict[int, list] = {}
            for s in pos_sigs:
                if s["scores"].get(aid, 0.0) < 0.5:
                    continue
                chunk_to_sigs.setdefault(s["chunk_idx"], []).append(s)
            if not chunk_to_sigs:
                continue

            spread_data: list[tuple] = []
            for ci_s in sorted(chunk_to_sigs):
                c_sigs = sorted(chunk_to_sigs[ci_s], key=lambda s: s.get("offset", 50))
                cx = float(x_dense[ci_s])
                lo = max(0.0,   cx - chunk_width * 0.5 + margin)
                hi = min(100.0, cx + chunk_width * 0.5 - margin)
                for sig in c_sigs:
                    off = sig.get("offset", 50) / 100.0
                    xi  = lo + off * (hi - lo)
                    spread_data.append((xi, ci_s, sig))

            xs_arr = np.array([d[0] for d in spread_data])
            ys_arr = np.array([d[2]["scores"].get(aid, 0.0) for d in spread_data])
            n_sig  = len(ys_arr)
            ys_smooth = np.array([
                0.6 * ys_arr[i]
                + 0.1 * ys_arr[max(0, i - 1)]
                + 0.1 * ys_arr[min(n_sig - 1, i + 1)]
                + 0.2 * chunk_sentiment.get(spread_data[i][1], 0.0)
                for i in range(n_sig)
            ])

            if np.abs(ys_smooth).max() < 0.1:
                continue

            net_fine = np.interp(x_fine, xs_arr, ys_smooth,
                                 left=ys_smooth[0], right=ys_smooth[-1])

            first_ci  = spread_data[0][1]
            last_ci   = spread_data[-1][1]
            x_start   = max(0.0,   float(x_dense[first_ci]) - chunk_width * 0.5)
            x_end     = min(100.0, float(x_dense[last_ci])  + chunk_width * 0.5)
            line_mask = (x_fine >= x_start) & (x_fine <= x_end)

            ax.plot(x_fine[line_mask], net_fine[line_mask], color=col, linewidth=1.8,
                    label=sheet.name, alpha=0.92, solid_capstyle="round", zorder=3)
            ax_yvals.extend(net_fine[line_mask].tolist())
            any_line = True

            for xi, ci_s, sig in spread_data:
                yi = float(np.interp(xi, x_fine, net_fine))
                ax_yvals.append(yi)
                ax.plot(xi, yi, "o", color=col, markersize=4.5,
                        markeredgewidth=0.8, markeredgecolor="white", alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=yi, aid=aid, aname=aname,
                                   chunk_idx=ci_s, sheet=sheet, signal=sig))

        if not any_line:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if ax_yvals:
            ax.set_ylim(min(ax_yvals) - 0.5, max(ax_yvals) + 0.5)
        ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
        ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        ax.set_ylabel("Emotion", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BORDER)
        if any_line:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8,
                      facecolor=BG, edgecolor=BORDER, handlelength=1.4)

    fig.tight_layout(pad=0.9, h_pad=1.4)
    return fig, marked


def _make_emotion_by_group_plot_widget(parent, sheets, total_chunks=None):
    """Interactive by-group emotion line-plot widget."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_emotion_by_group_fig(sheets, total_chunks=total_chunks)
    if fig is None:
        return None

    frame     = tk.Frame(parent, bg=BG_RESULTS)
    canvas    = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1); yr = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        tk_widget.config(cursor="hand2" if best and best_dist < 0.01 else "")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sig   = point["signal"]
        aid   = point["aid"]
        popup = tk.Toplevel(parent)
        popup.title(point["aname"])
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()
        tk.Label(popup, text=f"{point['sheet'].name}  ·  {point['aname']}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))
        body = tk.Frame(popup, bg=BG)
        body.pack(fill=tk.X, padx=24, pady=(0, 6))
        tk.Label(body, text=sig.get("signal", ""), font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=400, justify=tk.LEFT, anchor="nw").pack(fill=tk.X, expand=True)
        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))
        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root   = parent.winfo_toplevel()
        popup.geometry(f"+{root.winfo_rootx() + (root.winfo_width()  - pw) // 2}"
                       f"+{root.winfo_rooty() + (root.winfo_height() - ph) // 2}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)
    return frame


def _render_emotion_by_group_plot(sheets):
    """Render by-group emotion time-series to a PIL Image (for PDF embedding)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None
    fig, _ = _build_emotion_by_group_fig(sheets)
    if fig is None:
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _make_combined_personality_radar_widget(parent, sheets, extra_traits=None) -> "tk.Frame | None":
    """
    Render a radar chart with all characters overlaid (one polygon per character).
    Uses |avg| for spoke length, dimension name for labels.
    Returns a tk.Frame or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None

    traits = load_personality_traits()
    if extra_traits:
        known_ids = {t["id"] for t in traits}
        traits = traits + [t for t in extra_traits if t["id"] not in known_ids]
    if not traits:
        return None

    all_ids = [t["id"] for t in traits]
    labels  = [t["name"] for t in traits]
    N = len(traits)
    angles  = [n / float(N) * 2 * 3.14159265 for n in range(N)]

    FIG_BG = "#fdf9f6"
    AX_BG  = "#faf5ec"
    BORD   = "#d4ba90"

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    all_values = []
    char_data  = []
    for sheet in chars:
        totals = {tid: 0.0 for tid in all_ids}
        T = sheet.total_signals or 1
        for sig in sheet.signals_ordered:
            for tid, score in sig.get("scores", {}).items():
                if tid in totals:
                    totals[tid] += score
        avg     = [totals[tid] / T for tid in all_ids]
        abs_avg = [abs(v) for v in avg]
        floor   = max(abs_avg) * 0.06 if max(abs_avg) > 0 else 0
        values  = [max(v, floor) for v in abs_avg]
        all_values.extend(values)
        char_data.append((sheet.name, values))

    if not all_values or max(all_values) < 0.05:
        plt.close(fig)
        return None

    import numpy as np
    max_v = max(all_values)
    ticks = np.linspace(0, max_v, 4)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_ylim(0, max_v * 1.18)
    ax.yaxis.grid(color=BORD, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.grid(color=BORD, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(BORD)

    for ci, (cname, values) in enumerate(char_data):
        col    = CHAR_COLORS[ci % len(CHAR_COLORS)]
        v_plot = values + [values[0]]
        a_plot = angles + [angles[0]]
        ax.fill(a_plot, v_plot, color=col, alpha=0.18)
        ax.plot(a_plot, v_plot, color=col, linewidth=1.6, label=cname)
        ax.scatter(angles, values, color=col, s=14, zorder=4, edgecolors="white", linewidths=0.6)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.5, color="#111111")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.32, 1.15),
              framealpha=0.85, facecolor=FIG_BG, edgecolor=BORD)

    fig.tight_layout(pad=0.3)

    frame  = tk.Frame(parent, bg=BG)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    return frame


def _build_personality_by_group_fig(sheets, total_chunks=None):
    """
    Personality by-group time-series figure.
    One subplot per trait (top_archetypes[:3] across all characters).
    One line per character, signed y-values, zero baseline.
    Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chars = [s for s in sheets if s.signals_ordered]
    if not chars:
        return None, []

    # Collect unique trait IDs from top_archetypes[:3] across all chars
    trait_ids_ordered = []
    seen = set()
    for sheet in chars:
        for tid, tname, _ in sheet.top_archetypes[:3]:
            if tid not in seen:
                seen.add(tid)
                trait_ids_ordered.append((tid, tname))

    if not trait_ids_ordered:
        return None, []

    all_cidxs = [s["chunk_idx"] for sh in chars for s in sh.signals_ordered]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n_trait = len(trait_ids_ordered)
    fig, axes = plt.subplots(n_trait, 1, figsize=(9, max(2.4, n_trait * 2.2)),
                              facecolor=BG, squeeze=False)
    axes  = axes[:, 0]
    marked = []

    x_dense     = np.linspace(0, 100, total_chunks)
    chunk_width = 100.0 / max(total_chunks - 1, 1)
    margin      = chunk_width * 0.25
    n_fine      = total_chunks * 4
    x_fine      = np.linspace(0, 100, n_fine)

    for ax, (tid, tname) in zip(axes, trait_ids_ordered):
        ax.set_facecolor("#faf5ec")
        ax.set_title(tname, fontsize=9, color=TEXT, loc="left", pad=4)
        ax_yvals = []
        any_line = False

        for ci_char, sheet in enumerate(chars):
            col      = CHAR_COLORS[ci_char % len(CHAR_COLORS)]
            pos_sigs = sheet.signals_ordered

            chunk_to_sigs: dict[int, list] = {}
            for s in pos_sigs:
                if abs(s["scores"].get(tid, 0.0)) < 0.5:
                    continue
                chunk_to_sigs.setdefault(s["chunk_idx"], []).append(s)
            if not chunk_to_sigs:
                continue

            spread_data: list[tuple] = []
            for ci_s in sorted(chunk_to_sigs):
                c_sigs = sorted(chunk_to_sigs[ci_s], key=lambda s: s.get("offset", 50))
                cx = float(x_dense[ci_s])
                lo = max(0.0,   cx - chunk_width * 0.5 + margin)
                hi = min(100.0, cx + chunk_width * 0.5 - margin)
                for sig in c_sigs:
                    off = sig.get("offset", 50) / 100.0
                    xi  = lo + off * (hi - lo)
                    spread_data.append((xi, ci_s, sig))

            if not spread_data:
                continue

            xs_arr = np.array([d[0] for d in spread_data])
            ys_arr = np.array([d[2]["scores"].get(tid, 0.0) for d in spread_data])

            if np.max(np.abs(ys_arr)) < 0.1:
                continue

            net_fine = np.interp(x_fine, xs_arr, ys_arr,
                                 left=ys_arr[0], right=ys_arr[-1])

            first_ci  = spread_data[0][1]
            last_ci   = spread_data[-1][1]
            x_start   = max(0.0,   float(x_dense[first_ci]) - chunk_width * 0.5)
            x_end     = min(100.0, float(x_dense[last_ci])  + chunk_width * 0.5)
            line_mask = (x_fine >= x_start) & (x_fine <= x_end)

            ax.plot(x_fine[line_mask], net_fine[line_mask], color=col, linewidth=1.8,
                    label=sheet.name, alpha=0.92, solid_capstyle="round", zorder=3)
            ax_yvals.extend(net_fine[line_mask].tolist())
            any_line = True

            for xi, ci_s, sig in spread_data:
                yi = float(np.interp(xi, x_fine, net_fine))
                ax_yvals.append(yi)
                ax.plot(xi, yi, "o", color=col, markersize=4.5,
                        markeredgewidth=0.8, markeredgecolor="white", alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=yi, aid=tid, aname=tname,
                                   chunk_idx=ci_s, sheet=sheet, signal=sig))

        if not any_line:
            ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8)

        ax.axhline(0, color=BORDER, linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if ax_yvals:
            ylo = min(ax_yvals + [-0.2])
            yhi = max(ax_yvals + [0.2])
            pad = (yhi - ylo) * 0.15 or 0.3
            ax.set_ylim(ylo - pad, yhi + pad)
        ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
        ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        ax.set_ylabel("← low  ·  high →", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BORDER)
        if any_line:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8,
                      facecolor=BG, edgecolor=BORDER, handlelength=1.4)

    fig.tight_layout(pad=0.9, h_pad=1.4)
    return fig, marked


def _make_personality_by_group_plot_widget(parent, sheets, total_chunks=None):
    """Interactive by-group personality trait line-plot widget."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_personality_by_group_fig(sheets, total_chunks=total_chunks)
    if fig is None:
        return None

    frame     = tk.Frame(parent, bg=BG_RESULTS)
    canvas    = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1); yr = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        tk_widget.config(cursor="hand2" if best and best_dist < 0.01 else "")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sig   = point["signal"]
        aid   = point["aid"]
        popup = tk.Toplevel(parent)
        popup.title(point["aname"])
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()
        tk.Label(popup, text=f"{point['sheet'].name}  ·  {point['aname']}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))
        body = tk.Frame(popup, bg=BG)
        body.pack(fill=tk.X, padx=24, pady=(0, 6))
        tk.Label(body, text=sig.get("signal", ""), font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=400, justify=tk.LEFT, anchor="nw").pack(fill=tk.X)
        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))
        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root   = parent.winfo_toplevel()
        popup.geometry(f"+{root.winfo_rootx() + (root.winfo_width()  - pw) // 2}"
                       f"+{root.winfo_rooty() + (root.winfo_height() - ph) // 2}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)
    return frame


def _render_personality_by_group_plot(sheets):
    """Render by-group personality trait time-series to a PIL Image (for PDF embedding)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image as PILImage
    except ImportError:
        return None
    fig, _ = _build_personality_by_group_fig(sheets)
    if fig is None:
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _write_personality_report(path: str, title: str, sheets, group_by: str = "character") -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.lib.utils import simpleSplit

    C_BG      = HexColor("#fdf9f6")
    C_ACCENT  = HexColor("#c9a96e")
    C_ACCENT2 = HexColor("#b8954e")
    C_BAR_BG  = HexColor("#ede0c8")
    C_TEXT    = HexColor("#111111")
    C_MUTED   = HexColor("#6b7280")
    C_BLUE    = HexColor("#1e40af")
    C_BORDER  = HexColor("#d4ba90")
    C_WHITE   = white

    W, H = A4
    MARGIN    = 20 * mm
    CONTENT_W = W - 2 * MARGIN
    BAND_H    = 18 * mm
    FOOTER_H  = 10 * mm
    TOP_Y     = H - BAND_H - 6 * mm

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm
    page_num  = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror — {title} — Personality Analysis")

    # pole-name lookup for bipolar display
    from core.personality_analyzer import load_personality_traits as _lpt
    _pole_map_pdf = {
        t["id"]: (t.get("high_name", t["name"]), t.get("low_name", t["name"]))
        for t in _lpt()
    }

    def _draw_page_chrome():
        page_num[0] += 1
        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2)
        c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)

        band_mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, band_mid_y, "Personality Analysis")

        try:
            logo_x = W - MARGIN - logo_size
            logo_y = H - BAND_H + (BAND_H - logo_size) / 2
            c.drawImage(logo_path, logo_x, logo_y, width=logo_size, height=logo_size, mask="auto")
        except Exception:
            pass

        c.setFont("Helvetica", 8)
        c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOTER_H / 2, f"Mirror · {page_num[0]}")

    def new_page():
        c.showPage()
        _draw_page_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOTER_H + 4 * mm:
            new_page()
            return TOP_Y
        return y

    _draw_page_chrome()
    y = TOP_Y

    c.setFont("Helvetica", 10)
    c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 8 * mm

    if group_by == "trait":
        # ── By-Trait view: combined radar + by-trait sections ──────────────────
        from io import BytesIO as _BIO
        from reportlab.lib.utils import ImageReader as _IR
        import matplotlib
        matplotlib.use("Agg")

        # Render combined radar
        _radar_img = None
        try:
            import matplotlib.pyplot as _plt
            import numpy as _np

            traits = load_personality_traits()
            all_ids   = [t["id"] for t in traits]
            labels_r  = [t["name"] for t in traits]
            N_r = len(traits)
            angles_r = [n / float(N_r) * 2 * 3.14159265 for n in range(N_r)]

            fig_r, ax_r = _plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#fdf9f6")
            ax_r.set_facecolor("#faf5ec")
            all_vals_r = []
            char_data_r = []
            for sheet in sheets:
                totals = {tid: 0.0 for tid in all_ids}
                T = sheet.total_signals or 1
                for sig in sheet.signals_ordered:
                    for tid, score in sig.get("scores", {}).items():
                        if tid in totals:
                            totals[tid] += score
                avg     = [totals[tid] / T for tid in all_ids]
                abs_avg = [abs(v) for v in avg]
                floor   = max(abs_avg) * 0.06 if max(abs_avg) > 0 else 0
                vals    = [max(v, floor) for v in abs_avg]
                all_vals_r.extend(vals)
                char_data_r.append((sheet.name, vals))
            if all_vals_r and max(all_vals_r) > 0.05:
                max_vr = max(all_vals_r)
                tks = _np.linspace(0, max_vr, 4)[1:]
                ax_r.set_yticks(tks); ax_r.set_yticklabels([])
                ax_r.set_ylim(0, max_vr * 1.18)
                ax_r.yaxis.grid(color="#d4ba90", linewidth=0.5, linestyle="--", alpha=0.7)
                ax_r.xaxis.grid(color="#d4ba90", linewidth=0.5, alpha=0.5)
                ax_r.spines["polar"].set_color("#d4ba90")
                for ci, (cname, vals) in enumerate(char_data_r):
                    col = CHAR_COLORS[ci % len(CHAR_COLORS)]
                    vp  = vals + [vals[0]]; ap = angles_r + [angles_r[0]]
                    ax_r.fill(ap, vp, color=col, alpha=0.18)
                    ax_r.plot(ap, vp, color=col, linewidth=1.6, label=cname)
                    ax_r.scatter(angles_r, vals, color=col, s=14, zorder=4, edgecolors="white", linewidths=0.6)
                ax_r.set_xticks(angles_r); ax_r.set_xticklabels(labels_r, fontsize=7.5, color="#111111")
                ax_r.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.32, 1.15),
                            framealpha=0.85, facecolor="#fdf9f6", edgecolor="#d4ba90")
                fig_r.tight_layout(pad=0.3)
                _buf_r = _BIO()
                fig_r.savefig(_buf_r, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
                _plt.close(fig_r)
                _buf_r.seek(0)
                from PIL import Image as _PILImage
                _radar_img = _PILImage.open(_buf_r).copy()
        except Exception:
            pass

        if _radar_img is not None:
            _rw = CONTENT_W
            _rh = _rw * _radar_img.height / _radar_img.width
            y = ensure_space(y, _rh + 4 * mm)
            _buf2 = _BIO(); _radar_img.save(_buf2, format="PNG"); _buf2.seek(0)
            c.drawImage(_IR(_buf2), MARGIN, y - _rh, width=_rw, height=_rh)
            y -= _rh + 8 * mm

        # Render by-trait combined line plot
        _plot_img = _render_personality_by_group_plot(sheets)
        if _plot_img is not None:
            _pw = CONTENT_W
            _ph = _pw * _plot_img.height / _plot_img.width
            y = ensure_space(y, _ph + 4 * mm)
            _buf3 = _BIO(); _plot_img.save(_buf3, format="PNG"); _buf3.seek(0)
            c.drawImage(_IR(_buf3), MARGIN, y - _ph, width=_pw, height=_ph)
            y -= _ph + 8 * mm

        # Evidence per trait: gather from all characters
        trait_ids_ordered = []
        seen_t = set()
        for sheet in sheets:
            for tid, tname, _ in sheet.top_archetypes[:5]:
                if tid not in seen_t:
                    seen_t.add(tid)
                    trait_ids_ordered.append((tid, tname))

        ICON_H_PDF = 5 * mm
        for tid, tname in trait_ids_ordered:
            any_quotes = any(sheet.evidence.get(tid) for sheet in sheets)
            if not any_quotes:
                continue
            y = ensure_space(y, 16 * mm)
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(C_BLUE)
            c.drawString(MARGIN, y, tname.upper())
            y -= 5 * mm
            for sheet in sheets:
                quotes = sheet.evidence.get(tid, [])
                if not quotes:
                    continue
                y = ensure_space(y, 10 * mm)
                c.setFont("Helvetica-Bold", 8)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN + 3 * mm, y, sheet.name)
                y -= 4 * mm
                for q in quotes:
                    wrapped = simpleSplit('"' + q + '"', "Helvetica-Oblique", 9, CONTENT_W - 8 * mm)
                    for ln in wrapped:
                        y = ensure_space(y, 4.5 * mm)
                        c.setFont("Helvetica-Oblique", 9)
                        c.setFillColor(C_TEXT)
                        c.drawString(MARGIN + 6 * mm, y, ln)
                        y -= 4 * mm
                    y -= 1 * mm
            y -= 2 * mm

        c.save()
        return

    # ── By-Character view (default) ────────────────────────────────────────────
    for idx, sheet in enumerate(sheets):
        if idx == 0:
            y = ensure_space(y, 60 * mm)
        else:
            new_page()
            y = TOP_Y

        c.setFillColor(C_TEXT)
        c.setFont("Helvetica-Bold", 16)
        name_text = sheet.name
        alias_str = ("  (" + ", ".join(sheet.aliases) + ")") if sheet.aliases else ""
        c.drawString(MARGIN, y, name_text)
        if alias_str:
            name_w = c.stringWidth(name_text, "Helvetica-Bold", 16)
            c.setFont("Helvetica", 10)
            c.setFillColor(C_MUTED)
            c.drawString(MARGIN + name_w, y, alias_str)
        y -= 6 * mm

        if sheet.synopsis:
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColor(C_MUTED)
            for ln in simpleSplit(sheet.synopsis, "Helvetica-Oblique", 10, CONTENT_W):
                y = ensure_space(y, 5 * mm)
                c.drawString(MARGIN, y, ln)
                y -= 4.5 * mm
            y -= 1 * mm

        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(MARGIN, y, f"{sheet.total_signals} personality signals")
        y -= 6 * mm

        top3 = sheet.top_archetypes[:3]
        if top3:
            import numpy as _np
            from io import BytesIO as _BIO
            from reportlab.lib.utils import ImageReader as _IR

            row_h_b    = 6.5 * mm
            bar_h_b    = 3.5 * mm
            chart_h_b  = len(top3) * row_h_b
            badge_h_b  = chart_h_b
            label_w_b  = 38 * mm
            score_w_b  = 14 * mm
            gap_b      = 4 * mm
            badge_strip = len(top3) * 20 * mm
            bar_w_b    = CONTENT_W - label_w_b - score_w_b - gap_b - badge_strip
            bar_x_b    = MARGIN + label_w_b
            score_x_b  = bar_x_b + bar_w_b + 3 * mm
            badge_x0   = bar_x_b + bar_w_b + score_w_b + gap_b
            max_score  = max(abs(s) for _, _, s in top3) or 1

            y = ensure_space(y, chart_h_b + 4 * mm)
            bar_start_y = y

            for tid, name, score in top3:
                bar_mid = y - row_h_b / 2
                bar_top = bar_mid + bar_h_b / 2
                fill_w  = bar_w_b * abs(score) / max_score
                high_n, low_n = _pole_map_pdf.get(tid, (name, name))
                pole_label = high_n if score >= 0 else low_n
                c.setFont("Helvetica", 9)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN, bar_mid - 1.5 * mm, pole_label)
                c.setFillColor(C_BAR_BG)
                c.rect(bar_x_b, bar_top - bar_h_b, bar_w_b, bar_h_b, fill=1, stroke=0)
                if fill_w > 0:
                    c.setFillColor(C_ACCENT)
                    c.rect(bar_x_b, bar_top - bar_h_b, fill_w, bar_h_b, fill=1, stroke=0)
                c.setFont("Helvetica", 8)
                c.setFillColor(C_MUTED)
                c.drawString(score_x_b, bar_mid - 1.5 * mm, f"{score:+.2f}")
                y -= row_h_b

            bx = badge_x0
            for trait_id, _, t_score in top3:
                try:
                    ic = Image.open(_personality_icon_path(trait_id, t_score)).convert("RGBA")
                    ow, oh = ic.size
                    bw = badge_h_b * ow / oh
                    buf = _BIO(); ic.save(buf, format="PNG"); buf.seek(0)
                    c.drawImage(_IR(buf), bx, bar_start_y - badge_h_b,
                                width=bw, height=badge_h_b, mask="auto")
                    bx += bw + 1 * mm
                except Exception:
                    pass

            y -= 4 * mm

        _plot_img = _render_personality_plot([sheet])
        if _plot_img is not None:
            from reportlab.lib.utils import ImageReader as _IR
            from io import BytesIO as _BIO
            _buf = _BIO(); _plot_img.save(_buf, format="PNG"); _buf.seek(0)
            _pw = CONTENT_W
            _ph = _pw * _plot_img.height / _plot_img.width
            y = ensure_space(y, _ph + 4 * mm)
            c.drawImage(_IR(_buf), MARGIN, y - _ph, width=_pw, height=_ph)
            y -= _ph + 6 * mm

        if sheet.top_archetypes:
            y -= 3 * mm

        ICON_H_PDF = 5 * mm
        for tid, trait_name, t_score in sheet.top_archetypes[:5]:
            quotes = sheet.evidence.get(tid, [])
            if not quotes:
                continue
            y = ensure_space(y, 14 * mm)
            _icon_drawn = False
            try:
                from io import BytesIO as _BIO2
                from reportlab.lib.utils import ImageReader as _IR2
                _ic2 = Image.open(_personality_icon_path(tid, t_score)).convert("RGBA")
                _buf2 = _BIO2(); _ic2.save(_buf2, format="PNG"); _buf2.seek(0)
                _iw2 = ICON_H_PDF * _ic2.width / _ic2.height
                c.drawImage(_IR2(_buf2), MARGIN, y - ICON_H_PDF + 1 * mm,
                            width=_iw2, height=ICON_H_PDF, mask="auto")
                _icon_drawn = True
            except Exception:
                pass
            _name_x = MARGIN + (ICON_H_PDF + 2 * mm if _icon_drawn else 0)
            _high_n, _low_n = _pole_map_pdf.get(tid, (trait_name, trait_name))
            _pole_label = _high_n if t_score >= 0 else _low_n
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(C_BLUE)
            c.drawString(_name_x, y, _pole_label.upper())
            y -= 4.5 * mm
            for q in quotes:
                wrapped = simpleSplit(
                    '“' + q + '”', "Helvetica-Oblique", 9, CONTENT_W - 6 * mm
                )
                for ln in wrapped:
                    y = ensure_space(y, 4.5 * mm)
                    c.setFont("Helvetica-Oblique", 9)
                    c.setFillColor(C_TEXT)
                    c.drawString(MARGIN + 3 * mm, y, ln)
                    y -= 4 * mm
                y -= 1 * mm
            y -= 2 * mm

        y = ensure_space(y, 8 * mm)
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.4)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 8 * mm

    c.save()


def _make_genre_pie_widget(parent, results: list[GenreResult]):
    """Radial bar chart of genre signal distribution."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None
    if not results:
        return None

    PALETTE = [
        "#c9a96e", "#8c6230", "#d4ba90", "#a07840", "#e8c878",
        "#7c5c1e", "#daa560", "#b8954e", "#f0d090", "#6b4a18",
        "#e0c060", "#c48040", "#f5e0a0", "#9a7030", "#d0a050",
    ]

    total = sum(r.total_signals for r in results)
    main  = [r for r in results if r.trope_signals and r.total_signals / total >= 0.08]
    other = [r for r in results if r.trope_signals and r.total_signals / total <  0.08]

    if not main:
        return None

    labels = [r.genre for r in main]
    sizes  = [r.total_signals for r in main]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(main))]

    n = len(labels)
    angles    = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bar_width = 2 * np.pi / n * 0.72
    max_size  = max(sizes)

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection="polar"),
                           facecolor=BG)
    ax.set_facecolor("#faf5ec")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.bar(angles, sizes, width=bar_width, bottom=0,
           color=colors, alpha=0.88, linewidth=0.5, edgecolor=BG, zorder=3)

    label_r = max_size * 1.1
    for angle, label, size in zip(angles, labels, sizes):
        x_cart = np.sin(angle)
        ha  = "left" if x_cart > 0.1 else ("right" if x_cart < -0.1 else "center")
        pct = f"{size / total * 100:.0f}%"
        ax.text(angle, label_r, f"{label}\n{pct}",
                ha=ha, va="center", fontsize=8, color=TEXT, linespacing=1.3)

    if other:
        names = ", ".join(f"{r.genre} ({r.total_signals/total*100:.0f}%)" for r in other)
        fig.text(0.5, 0.01, f"Other: {names}",
                 ha="center", fontsize=6, color=TEXT_MUTED, style="italic")

    ax.set_ylim(0, max_size * 1.55)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.yaxis.grid(True, linewidth=0.4, color=BORDER, alpha=0.5)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(f"{total} genre signals detected", fontsize=8.5,
                 color="#6b7280", pad=12)
    fig.tight_layout(pad=0.6)

    frame  = tk.Frame(parent, bg=BG)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.X)
    plt.close(fig)
    return frame


def _build_genre_fig(results: list, total_chunks=None):
    """
    Genre scatter plot: x = narrative progress, y = genre (categorical).
    Each signal is an 'x' marker. Returns (fig, marked) or (None, []).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    total  = sum(r.total_signals for r in results)
    genres = [r for r in results if r.trope_signals and r.total_signals / total >= 0.08]
    other  = [r for r in results if r.trope_signals and r.total_signals / total <  0.08]
    if not genres:
        return None, []

    LINE_COLORS = ["#c9a96e", "#7c5c1e", "#d4956a", "#b8954e",
                   "#e8c878", "#8c6230", "#daa560", "#9a7030",
                   "#d4ba90", "#a07840", "#f0d090", "#6b4a18"]

    all_cidxs = [s.chunk_idx for r in genres for s in r.trope_signals]
    if total_chunks is None:
        total_chunks = max(all_cidxs) + 1 if all_cidxs else 1

    n           = len(genres)
    fig_h       = max(2.5, n * 0.52 + 0.9)
    fig, ax     = plt.subplots(figsize=(9, fig_h), facecolor=BG)
    ax.set_facecolor("#faf5ec")

    x_dense     = np.linspace(0, 100, total_chunks)
    chunk_width = 100.0 / max(total_chunks - 1, 1)
    margin      = chunk_width * 0.25
    marked      = []

    for gi, (result, col) in enumerate(
        zip(genres, [LINE_COLORS[i % len(LINE_COLORS)] for i in range(n)])
    ):
        y_pos = n - 1 - gi   # top-ranked genre at the top

        chunk_to_sigs: dict[int, list] = {}
        for s in result.trope_signals:
            chunk_to_sigs.setdefault(s.chunk_idx, []).append(s)

        for ci_s, c_sigs in sorted(chunk_to_sigs.items()):
            cx = float(x_dense[ci_s])
            lo = max(0.0,   cx - chunk_width * 0.5 + margin)
            hi = min(100.0, cx + chunk_width * 0.5 - margin)
            xs = [cx] if len(c_sigs) == 1 else list(np.linspace(lo, hi, len(c_sigs)))
            for xi, sig in zip(xs, c_sigs):
                ax.plot(xi, y_pos, "x", color=col, markersize=6,
                        markeredgewidth=1.5, alpha=0.85, zorder=5)
                marked.append(dict(ax=ax, x=xi, y=float(y_pos),
                                   genre=result.genre, chunk_idx=ci_s, sig=sig))

    genre_names = [r.genre for r in reversed(genres)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(genre_names, fontsize=7.5)
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.5, n - 0.5)
    ax.tick_params(labelsize=7, colors=TEXT_MUTED, length=2)
    ax.set_xlabel("narrative progress", fontsize=6.5, color=TEXT_MUTED, labelpad=2)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)

    if other:
        names = ", ".join(f"{r.genre} ({r.total_signals/total*100:.0f}%)" for r in other)
        fig.text(0.5, 0.01, f"Other: {names}",
                 ha="center", fontsize=6, color=TEXT_MUTED, style="italic")

    fig.tight_layout(pad=0.9)
    return fig, marked


def _make_genre_plot_widget(parent, results: list, total_chunks=None):
    """Interactive genre line-plot widget. Click a dot to see the trope signal."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except ImportError:
        return None

    fig, marked = _build_genre_fig(results, total_chunks=total_chunks)
    if fig is None:
        return None

    frame     = tk.Frame(parent, bg=BG_RESULTS)
    canvas    = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk_widget = canvas.get_tk_widget()

    def _nearest(event):
        if event.inaxes is None or event.xdata is None:
            return None, float("inf")
        ax   = event.inaxes
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xr   = max(xlim[1] - xlim[0], 1); yr = max(ylim[1] - ylim[0], 1)
        best, best_dist = None, float("inf")
        for pt in marked:
            if pt["ax"] is not ax:
                continue
            dist = (((event.xdata - pt["x"]) / xr) ** 2 +
                    ((event.ydata - pt["y"]) / yr) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, pt
        return best, best_dist

    def on_motion(event):
        best, best_dist = _nearest(event)
        tk_widget.config(cursor="hand2" if best and best_dist < 0.01 else "")

    def on_leave(event):
        tk_widget.config(cursor="")

    def _show_signal_popup(point):
        sig   = point["sig"]   # TropeSignal
        popup = tk.Toplevel(parent)
        popup.title(point["genre"])
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(parent.winfo_toplevel())
        popup.grab_set()

        tk.Label(popup, text=f"{point['genre']}  ·  {sig.trope}",
                 font=FONT_LABEL, fg=TEXT, bg=BG).pack(padx=24, pady=(18, 6))
        tk.Label(popup, text=sig.signal, font=FONT_QUOTE, fg=TEXT, bg=BG,
                 wraplength=380, justify=tk.LEFT).pack(padx=24, pady=(0, 6))
        tk.Frame(popup, height=1, bg=BORDER).pack(fill=tk.X, padx=24, pady=(8, 0))
        _flat_btn(popup, "OK", popup.destroy, primary=True).pack(pady=(10, 18))

        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        root = parent.winfo_toplevel()
        x = root.winfo_rootx() + (root.winfo_width()  - pw) // 2
        y = root.winfo_rooty() + (root.winfo_height() - ph) // 2
        popup.geometry(f"+{x}+{y}")

    def on_click(event):
        best, best_dist = _nearest(event)
        if best and best_dist < 0.01:
            _show_signal_popup(best)

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("axes_leave_event",    on_leave)
    canvas.mpl_connect("button_press_event",  on_click)
    plt.close(fig)
    return frame


def _write_arch_report(path: str, title: str, sheets) -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import simpleSplit

    # ── colours (match app palette) ───────────────────────────
    C_BG      = HexColor("#fdf9f6")
    C_ACCENT  = HexColor("#c9a96e")
    C_ACCENT2 = HexColor("#b8954e")
    C_BAR_BG  = HexColor("#ede0c8")
    C_TEXT    = HexColor("#111111")
    C_MUTED   = HexColor("#6b7280")
    C_BLUE    = HexColor("#1e40af")
    C_BORDER  = HexColor("#d4ba90")
    C_WHITE   = white

    W, H = A4
    MARGIN    = 20 * mm
    CONTENT_W = W - 2 * MARGIN

    BAND_H   = 18 * mm
    FOOTER_H = 10 * mm
    TOP_Y    = H - BAND_H - 6 * mm

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm

    page_num = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror — {title}")

    def _draw_page_chrome():
        page_num[0] += 1

        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)

        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2)
        c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)

        # heading inside the band
        band_mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, band_mid_y, "Archetype Analysis")

        try:
            logo_x = W - MARGIN - logo_size
            logo_y = H - BAND_H + (BAND_H - logo_size) / 2
            c.drawImage(logo_path, logo_x, logo_y,
                        width=logo_size, height=logo_size,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass

        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.line(MARGIN, FOOTER_H, W - MARGIN, FOOTER_H)
        c.setFont("Helvetica", 8)
        c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOTER_H - 4 * mm, f"Page {page_num[0]}")

    def new_page():
        c.showPage()
        _draw_page_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOTER_H + 4 * mm:
            new_page()
            return TOP_Y
        return y

    # ── first page ────────────────────────────────────────────
    _draw_page_chrome()

    y = TOP_Y

    # subtitle: file name + date
    c.setFont("Helvetica", 10)
    c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 8 * mm

    for idx, sheet in enumerate(sheets):
        if idx == 0:
            y = ensure_space(y, 60 * mm)
        else:
            new_page()
            y = TOP_Y

        # ── character name ─────────────────────────────────────
        c.setFillColor(C_TEXT)
        c.setFont("Helvetica-Bold", 16)
        name_text = sheet.name
        alias_str = ("  (" + ", ".join(sheet.aliases) + ")") if sheet.aliases else ""
        c.drawString(MARGIN, y, name_text)
        if alias_str:
            name_w = c.stringWidth(name_text, "Helvetica-Bold", 16)
            c.setFont("Helvetica", 10)
            c.setFillColor(C_MUTED)
            c.drawString(MARGIN + name_w, y, alias_str)
        y -= 6 * mm

        # synopsis (wrapped)
        if sheet.synopsis:
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColor(C_MUTED)
            for ln in simpleSplit(sheet.synopsis, "Helvetica-Oblique", 10, CONTENT_W):
                y = ensure_space(y, 5 * mm)
                c.drawString(MARGIN, y, ln)
                y -= 4.5 * mm
            y -= 1 * mm

        # signal count
        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(MARGIN, y, f"{sheet.total_signals} signal instances")
        y -= 6 * mm

        # ── bar chart + badge icons ────────────────────────────
        top3 = sheet.top_archetypes[:3]
        if top3:
            import numpy as _np
            from io import BytesIO as _BIO
            from reportlab.lib.utils import ImageReader as _IR

            row_h_b    = 6.5 * mm
            bar_h_b    = 3.5 * mm
            chart_h_b  = len(top3) * row_h_b
            badge_h_b  = chart_h_b
            label_w_b  = 38 * mm
            score_w_b  = 14 * mm
            gap_b      = 4 * mm
            badge_strip = len(top3) * 20 * mm
            bar_w_b    = CONTENT_W - label_w_b - score_w_b - gap_b - badge_strip
            bar_x_b    = MARGIN + label_w_b
            score_x_b  = bar_x_b + bar_w_b + 3 * mm
            badge_x0   = bar_x_b + bar_w_b + score_w_b + gap_b
            max_score  = max(s for _, _, s in top3) or 1

            y = ensure_space(y, chart_h_b + 4 * mm)
            bar_start_y = y

            for _, name, score in top3:
                bar_mid = y - row_h_b / 2
                bar_top = bar_mid + bar_h_b / 2
                fill_w  = bar_w_b * score / max_score
                c.setFont("Helvetica", 9)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN, bar_mid - 1.5 * mm, name)
                c.setFillColor(C_BAR_BG)
                c.rect(bar_x_b, bar_top - bar_h_b, bar_w_b, bar_h_b, fill=1, stroke=0)
                if fill_w > 0:
                    c.setFillColor(C_ACCENT)
                    c.rect(bar_x_b, bar_top - bar_h_b, fill_w, bar_h_b, fill=1, stroke=0)
                c.setFont("Helvetica", 8)
                c.setFillColor(C_MUTED)
                c.drawString(score_x_b, bar_mid - 1.5 * mm, f"{score:.2f}")
                y -= row_h_b

            bx = badge_x0
            bg_rgb_b = (253, 249, 246)
            for arch_id, _, _ in top3:
                try:
                    _raw = Image.open(ARCH_ICON_PATH(arch_id)).convert("RGBA")
                    _bg  = Image.new("RGBA", _raw.size, bg_rgb_b + (255,))
                    ic   = Image.alpha_composite(_bg, _raw).convert("RGB")
                    arr  = _np.array(ic)
                    arr[(arr[:,:,0]>240)&(arr[:,:,1]>240)&(arr[:,:,2]>240)] = bg_rgb_b
                    ic = Image.fromarray(arr.astype(_np.uint8))
                    ow, oh = ic.size
                    bw = badge_h_b * ow / oh
                    buf = _BIO(); ic.save(buf, format="PNG"); buf.seek(0)
                    c.drawImage(_IR(buf), bx, bar_start_y - badge_h_b,
                                width=bw, height=badge_h_b)
                    bx += bw + 1 * mm
                except Exception:
                    pass

            y -= 4 * mm

        # ── per-character time-series plot ─────────────────────
        _plot_img = _render_archetype_plot([sheet])
        if _plot_img is not None:
            from reportlab.lib.utils import ImageReader as _IR
            from io import BytesIO as _BIO
            _buf = _BIO(); _plot_img.save(_buf, format="PNG"); _buf.seek(0)
            _pw = CONTENT_W
            _ph = _pw * _plot_img.height / _plot_img.width
            y = ensure_space(y, _ph + 4 * mm)
            c.drawImage(_IR(_buf), MARGIN, y - _ph, width=_pw, height=_ph)
            y -= _ph + 6 * mm

        # ── evidence quotes ────────────────────────────────────
        for aid, arch_name, _ in sheet.top_archetypes[:5]:
            quotes = sheet.evidence.get(aid, [])
            if not quotes:
                continue
            y = ensure_space(y, 14 * mm)
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(C_BLUE)
            c.drawString(MARGIN, y, arch_name.upper())
            y -= 4.5 * mm
            for q in quotes:
                wrapped = simpleSplit(f"\u201c{q}\u201d", "Helvetica-Oblique", 9, CONTENT_W - 6 * mm)
                for ln in wrapped:
                    y = ensure_space(y, 4.5 * mm)
                    c.setFont("Helvetica-Oblique", 9)
                    c.setFillColor(C_TEXT)
                    c.drawString(MARGIN + 3 * mm, y, ln)
                    y -= 4 * mm
                y -= 1 * mm
            y -= 2 * mm

        # divider
        y = ensure_space(y, 8 * mm)
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.4)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 8 * mm

    c.save()


def _write_emotion_report(path: str, title: str, sheets, group_by: str = "character") -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.lib.utils import simpleSplit

    C_BG      = HexColor("#fdf9f6")
    C_ACCENT  = HexColor("#c9a96e")
    C_ACCENT2 = HexColor("#b8954e")
    C_BAR_BG  = HexColor("#ede0c8")
    C_TEXT    = HexColor("#111111")
    C_MUTED   = HexColor("#6b7280")
    C_BLUE    = HexColor("#1e40af")
    C_BORDER  = HexColor("#d4ba90")
    C_WHITE   = white

    W, H = A4
    MARGIN    = 20 * mm
    CONTENT_W = W - 2 * MARGIN
    BAND_H    = 18 * mm
    FOOTER_H  = 10 * mm
    TOP_Y     = H - BAND_H - 6 * mm

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm
    page_num  = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror — {title} — Emotion Analysis")

    def _draw_page_chrome():
        page_num[0] += 1
        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2)
        c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)

        # heading inside the band
        band_mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, band_mid_y, "Emotion Analysis")

        try:
            logo_x = W - MARGIN - logo_size
            logo_y = H - BAND_H + (BAND_H - logo_size) / 2
            c.drawImage(logo_path, logo_x, logo_y,
                        width=logo_size, height=logo_size,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.line(MARGIN, FOOTER_H, W - MARGIN, FOOTER_H)
        c.setFont("Helvetica", 8)
        c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOTER_H - 4 * mm, f"Page {page_num[0]}")

    def new_page():
        c.showPage()
        _draw_page_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOTER_H + 4 * mm:
            new_page()
            return TOP_Y
        return y

    _draw_page_chrome()
    y = TOP_Y

    # subtitle: file name + date
    c.setFont("Helvetica", 10)
    c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 8 * mm

    if group_by == "emotion":
        # ── By-Emotion view: combined radar + by-emotion sections ──────────────
        from io import BytesIO as _BIO
        from reportlab.lib.utils import ImageReader as _IR
        import matplotlib
        matplotlib.use("Agg")

        # Render combined radar
        _radar_img = None
        try:
            import matplotlib.pyplot as _plt
            import numpy as _np

            emotions   = load_emotions()
            by_id_e    = {e["id"]: e for e in emotions}
            ordered_e  = [by_id_e[eid] for eid in PLUTCHIK_ORDER if eid in by_id_e]
            ordered_e  += [e for e in emotions if e["id"] not in PLUTCHIK_ORDER]
            all_ids_e  = [e["id"] for e in ordered_e]
            labels_e   = [e["name"] for e in ordered_e]
            N_e        = len(ordered_e)
            angles_e   = _np.linspace(0, 2 * _np.pi, N_e, endpoint=False).tolist()

            fig_r, ax_r = _plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#fdf9f6")
            ax_r.set_facecolor("#faf5ec")
            all_vals_e  = []
            char_data_e = []
            for sheet in sheets:
                totals = {eid: 0.0 for eid in all_ids_e}
                T = sheet.total_signals or 1
                for sig in sheet.signals_ordered:
                    for eid, score in sig.get("scores", {}).items():
                        if eid in totals:
                            totals[eid] += score
                raw   = [totals[eid] / T for eid in all_ids_e]
                floor = max(raw) * 0.06 if max(raw) > 0 else 0
                vals  = [max(v, floor) for v in raw]
                all_vals_e.extend(vals)
                char_data_e.append((sheet.name, vals))
            if all_vals_e and max(all_vals_e) > 0.05:
                max_ve = max(all_vals_e)
                tks = _np.linspace(0, max_ve, 4)[1:]
                ax_r.set_yticks(tks); ax_r.set_yticklabels([])
                ax_r.set_ylim(0, max_ve * 1.18)
                ax_r.yaxis.grid(color="#d4ba90", linewidth=0.5, linestyle="--", alpha=0.7)
                ax_r.xaxis.grid(color="#d4ba90", linewidth=0.5, alpha=0.5)
                ax_r.spines["polar"].set_color("#d4ba90")
                for ci, (cname, vals) in enumerate(char_data_e):
                    col = CHAR_COLORS[ci % len(CHAR_COLORS)]
                    vp  = vals + [vals[0]]; ap = angles_e + [angles_e[0]]
                    ax_r.fill(ap, vp, color=col, alpha=0.18)
                    ax_r.plot(ap, vp, color=col, linewidth=1.6, label=cname)
                    ax_r.scatter(angles_e, vals, color=col, s=14, zorder=4, edgecolors="white", linewidths=0.6)
                ax_r.set_xticks(angles_e); ax_r.set_xticklabels(labels_e, fontsize=7.5, color="#111111")
                ax_r.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.32, 1.15),
                            framealpha=0.85, facecolor="#fdf9f6", edgecolor="#d4ba90")
                fig_r.tight_layout(pad=0.3)
                _buf_r = _BIO()
                fig_r.savefig(_buf_r, format="png", dpi=130, bbox_inches="tight", facecolor="#fdf9f6")
                _plt.close(fig_r)
                _buf_r.seek(0)
                from PIL import Image as _PILImage
                _radar_img = _PILImage.open(_buf_r).copy()
        except Exception:
            pass

        if _radar_img is not None:
            _rw = CONTENT_W
            _rh = _rw * _radar_img.height / _radar_img.width
            y = ensure_space(y, _rh + 4 * mm)
            _buf2 = _BIO(); _radar_img.save(_buf2, format="PNG"); _buf2.seek(0)
            c.drawImage(_IR(_buf2), MARGIN, y - _rh, width=_rw, height=_rh)
            y -= _rh + 8 * mm

        # Render by-emotion combined line plot
        _plot_img = _render_emotion_by_group_plot(sheets)
        if _plot_img is not None:
            _pw = CONTENT_W
            _ph = _pw * _plot_img.height / _plot_img.width
            y = ensure_space(y, _ph + 4 * mm)
            _buf3 = _BIO(); _plot_img.save(_buf3, format="PNG"); _buf3.seek(0)
            c.drawImage(_IR(_buf3), MARGIN, y - _ph, width=_pw, height=_ph)
            y -= _ph + 8 * mm

        # Evidence per emotion: gather from all characters
        emotion_ids_ordered = []
        seen_e = set()
        for sheet in sheets:
            for eid, ename, _ in sheet.top_archetypes[:5]:
                if eid not in seen_e:
                    seen_e.add(eid)
                    emotion_ids_ordered.append((eid, ename))

        ICON_H_PDF = 5 * mm
        for eid, ename in emotion_ids_ordered:
            any_quotes = any(sheet.evidence.get(eid) for sheet in sheets)
            if not any_quotes:
                continue
            y = ensure_space(y, 16 * mm)
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(C_BLUE)
            c.drawString(MARGIN, y, ename.upper())
            y -= 5 * mm
            for sheet in sheets:
                quotes = sheet.evidence.get(eid, [])
                if not quotes:
                    continue
                y = ensure_space(y, 10 * mm)
                c.setFont("Helvetica-Bold", 8)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN + 3 * mm, y, sheet.name)
                y -= 4 * mm
                for q in quotes:
                    wrapped = simpleSplit('"' + q + '"', "Helvetica-Oblique", 9, CONTENT_W - 8 * mm)
                    for ln in wrapped:
                        y = ensure_space(y, 4.5 * mm)
                        c.setFont("Helvetica-Oblique", 9)
                        c.setFillColor(C_TEXT)
                        c.drawString(MARGIN + 6 * mm, y, ln)
                        y -= 4 * mm
                    y -= 1 * mm
            y -= 2 * mm

        c.save()
        return

    # ── By-Character view (default) ────────────────────────────────────────────
    for idx, sheet in enumerate(sheets):
        if idx == 0:
            y = ensure_space(y, 60 * mm)
        else:
            new_page()
            y = TOP_Y

        c.setFillColor(C_TEXT)
        c.setFont("Helvetica-Bold", 16)
        name_text = sheet.name
        alias_str = ("  (" + ", ".join(sheet.aliases) + ")") if sheet.aliases else ""
        c.drawString(MARGIN, y, name_text)
        if alias_str:
            name_w = c.stringWidth(name_text, "Helvetica-Bold", 16)
            c.setFont("Helvetica", 10)
            c.setFillColor(C_MUTED)
            c.drawString(MARGIN + name_w, y, alias_str)
        y -= 6 * mm

        if sheet.synopsis:
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColor(C_MUTED)
            for ln in simpleSplit(sheet.synopsis, "Helvetica-Oblique", 10, CONTENT_W):
                y = ensure_space(y, 5 * mm)
                c.drawString(MARGIN, y, ln)
                y -= 4.5 * mm
            y -= 1 * mm

        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(MARGIN, y, f"{sheet.total_signals} emotional moments")
        y -= 6 * mm

        # ── bar chart + badge icons ────────────────────────────
        top3 = sheet.top_archetypes[:3]
        if top3:
            import numpy as _np
            from io import BytesIO as _BIO
            from reportlab.lib.utils import ImageReader as _IR

            row_h_b   = 6.5 * mm
            bar_h_b   = 3.5 * mm
            chart_h_b = len(top3) * row_h_b
            badge_h_b = chart_h_b
            label_w_b = 38 * mm
            score_w_b = 14 * mm
            gap_b     = 4 * mm
            badge_strip = len(top3) * 20 * mm
            bar_w_b   = CONTENT_W - label_w_b - score_w_b - gap_b - badge_strip
            bar_x_b   = MARGIN + label_w_b
            score_x_b = bar_x_b + bar_w_b + 3 * mm
            badge_x0  = bar_x_b + bar_w_b + score_w_b + gap_b
            max_score = max(s for _, _, s in top3) or 1

            y = ensure_space(y, chart_h_b + 4 * mm)
            bar_start_y = y

            for _, name, score in top3:
                bar_mid = y - row_h_b / 2
                bar_top = bar_mid + bar_h_b / 2
                fill_w  = bar_w_b * score / max_score
                c.setFont("Helvetica", 9)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN, bar_mid - 1.5 * mm, name)
                c.setFillColor(C_BAR_BG)
                c.rect(bar_x_b, bar_top - bar_h_b, bar_w_b, bar_h_b, fill=1, stroke=0)
                if fill_w > 0:
                    c.setFillColor(C_ACCENT)
                    c.rect(bar_x_b, bar_top - bar_h_b, fill_w, bar_h_b, fill=1, stroke=0)
                c.setFont("Helvetica", 8)
                c.setFillColor(C_MUTED)
                c.drawString(score_x_b, bar_mid - 1.5 * mm, f"{score:.2f}")
                y -= row_h_b

            bx = badge_x0
            for emot_id, _, _ in top3:
                try:
                    ic = Image.open(_emotion_icon_path(emot_id)).convert("RGBA")
                    ow, oh = ic.size
                    bw = badge_h_b * ow / oh
                    buf = _BIO(); ic.save(buf, format="PNG"); buf.seek(0)
                    c.drawImage(_IR(buf), bx, bar_start_y - badge_h_b,
                                width=bw, height=badge_h_b, mask="auto")
                    bx += bw + 1 * mm
                except Exception:
                    pass

            y -= 4 * mm

        # ── per-character time-series plot ─────────────────────
        _plot_img = _render_emotion_plot([sheet])
        if _plot_img is not None:
            from reportlab.lib.utils import ImageReader as _IR
            from io import BytesIO as _BIO
            _buf = _BIO(); _plot_img.save(_buf, format="PNG"); _buf.seek(0)
            _pw = CONTENT_W
            _ph = _pw * _plot_img.height / _plot_img.width
            y = ensure_space(y, _ph + 4 * mm)
            c.drawImage(_IR(_buf), MARGIN, y - _ph, width=_pw, height=_ph)
            y -= _ph + 6 * mm

        if sheet.top_archetypes:
            y -= 3 * mm

        ICON_H_PDF = 5 * mm
        for eid, emot_name, _ in sheet.top_archetypes[:5]:
            quotes = sheet.evidence.get(eid, [])
            if not quotes:
                continue
            y = ensure_space(y, 14 * mm)
            # small icon to the left of the emotion name
            _icon_drawn = False
            try:
                from io import BytesIO as _BIO2
                from reportlab.lib.utils import ImageReader as _IR2
                _ic2 = Image.open(_emotion_icon_path(eid)).convert("RGBA")
                _buf2 = _BIO2(); _ic2.save(_buf2, format="PNG"); _buf2.seek(0)
                _iw2 = ICON_H_PDF * _ic2.width / _ic2.height
                c.drawImage(_IR2(_buf2), MARGIN, y - ICON_H_PDF + 1 * mm,
                            width=_iw2, height=ICON_H_PDF, mask="auto")
                _icon_drawn = True
            except Exception:
                pass
            _name_x = MARGIN + (ICON_H_PDF + 2 * mm if _icon_drawn else 0)
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(C_BLUE)
            c.drawString(_name_x, y, emot_name.upper())
            y -= 4.5 * mm
            for q in quotes:
                wrapped = simpleSplit(
                    '"' + q + '"', "Helvetica-Oblique", 9, CONTENT_W - 6 * mm
                )
                for ln in wrapped:
                    y = ensure_space(y, 4.5 * mm)
                    c.setFont("Helvetica-Oblique", 9)
                    c.setFillColor(C_TEXT)
                    c.drawString(MARGIN + 3 * mm, y, ln)
                    y -= 4 * mm
                y -= 1 * mm
            y -= 2 * mm

        y = ensure_space(y, 8 * mm)
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.4)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 8 * mm

    c.save()


def _write_genre_report(path: str, title: str, results: list[GenreResult]) -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.lib.utils import simpleSplit

    C_BG     = HexColor("#fdf9f6")
    C_ACCENT = HexColor("#c9a96e")
    C_ACCENT2= HexColor("#b8954e")
    C_TEXT   = HexColor("#111111")
    C_MUTED  = HexColor("#6b7280")
    C_BORDER = HexColor("#d4ba90")
    C_HDR    = HexColor("#ede0c8")
    C_ROW0   = HexColor("#fdf9f6")
    C_ROW1   = HexColor("#ffffff")
    C_WHITE  = white

    W, H = A4
    MARGIN    = 20 * mm
    CONTENT_W = W - 2 * MARGIN
    BAND_H    = 18 * mm
    FOOTER_H  = 10 * mm
    TOP_Y     = H - BAND_H - 6 * mm

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm
    page_num  = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror — {title} — Genre Analysis")

    def _draw_page_chrome():
        page_num[0] += 1
        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2)
        c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)
        band_mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, band_mid_y, "Genre Analysis")
        try:
            logo_x = W - MARGIN - logo_size
            logo_y = H - BAND_H + (BAND_H - logo_size) / 2
            c.drawImage(logo_path, logo_x, logo_y,
                        width=logo_size, height=logo_size,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.5)
        c.line(MARGIN, FOOTER_H, W - MARGIN, FOOTER_H)
        c.setFont("Helvetica", 8)
        c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOTER_H - 4 * mm, f"Page {page_num[0]}")

    def new_page():
        c.showPage()
        _draw_page_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOTER_H + 4 * mm:
            new_page()
            return TOP_Y
        return y

    _draw_page_chrome()
    y = TOP_Y

    c.setFont("Helvetica", 10)
    c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 10 * mm

    # ── radial bar chart + line plot ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image as _PILImage
        from reportlab.lib.utils import ImageReader

        def _fig_to_image(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return _PILImage.open(buf).copy()

        def _embed(img, max_w, extra_gap=8):
            nonlocal y
            iw = min(max_w, CONTENT_W)
            ih = iw * img.height / img.width
            y  = ensure_space(y, ih + extra_gap * mm)
            b  = BytesIO(); img.save(b, format="PNG"); b.seek(0)
            c.drawImage(ImageReader(b), MARGIN + (CONTENT_W - iw) / 2,
                        y - ih, width=iw, height=ih)
            y -= ih + extra_gap * mm

        PALETTE = ["#c9a96e","#8c6230","#d4ba90","#a07840","#e8c878",
                   "#7c5c1e","#daa560","#b8954e","#f0d090","#6b4a18",
                   "#e0c060","#c48040","#f5e0a0","#9a7030","#d0a050"]

        total_sigs  = sum(r.total_signals for r in results)
        main_r  = [r for r in results if r.trope_signals and r.total_signals / total_sigs >= 0.08]
        other_r = [r for r in results if r.trope_signals and r.total_signals / total_sigs <  0.08]

        # Radial bar chart — genres ≥ 8%
        if main_r:
            labels = [r.genre for r in main_r]
            sizes  = [r.total_signals for r in main_r]
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(main_r))]
            n      = len(labels)
            angles    = np.linspace(0, 2 * np.pi, n, endpoint=False)
            bar_width = 2 * np.pi / n * 0.72
            max_size  = max(sizes)

            fig_r, ax_r = plt.subplots(figsize=(5, 5),
                                       subplot_kw=dict(projection="polar"),
                                       facecolor="#fdf9f6")
            ax_r.set_facecolor("#faf5ec")
            ax_r.set_theta_offset(np.pi / 2)
            ax_r.set_theta_direction(-1)
            ax_r.bar(angles, sizes, width=bar_width, bottom=0,
                     color=colors, alpha=0.88, linewidth=0.5,
                     edgecolor="#fdf9f6", zorder=3)
            label_r = max_size * 1.1
            for angle, label, size in zip(angles, labels, sizes):
                x_cart = np.sin(angle)
                ha  = "left" if x_cart > 0.1 else ("right" if x_cart < -0.1 else "center")
                pct = f"{size / total_sigs * 100:.0f}%"
                ax_r.text(angle, label_r, f"{label}\n{pct}",
                          ha=ha, va="center", fontsize=7, color="#111111",
                          linespacing=1.3)
            if other_r:
                names = ", ".join(
                    f"{r.genre} ({r.total_signals/total_sigs*100:.0f}%)" for r in other_r
                )
                fig_r.text(0.5, 0.01, f"Other: {names}",
                           ha="center", fontsize=6, color="#6b7280", style="italic")
            ax_r.set_ylim(0, max_size * 1.55)
            ax_r.set_yticklabels([])
            ax_r.set_xticks([])
            ax_r.yaxis.grid(True, linewidth=0.4, color="#d4ba90", alpha=0.5)
            ax_r.xaxis.grid(False)
            ax_r.spines["polar"].set_visible(False)
            ax_r.set_title(f"{total_sigs} genre signals detected", fontsize=8,
                           color="#6b7280", pad=12)
            fig_r.tight_layout(pad=0.6)
            _embed(_fig_to_image(fig_r), CONTENT_W * 0.7)

        # Top-3 line plot
        fig_l, marked_l = _build_genre_fig(results)
        if fig_l is not None:
            _embed(_fig_to_image(fig_l), CONTENT_W)

    except Exception:
        pass

    # ── top signals per genre ─────────────────────────────────────────────────
    for r in results:
        y = ensure_space(y, 20 * mm)
        total_pct = f"  {r.total_signals / sum(x.total_signals for x in results) * 100:.0f}%"
        c.setFillColor(C_TEXT)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN, y, r.genre)
        c.setFont("Helvetica", 9)
        c.setFillColor(C_MUTED)
        c.drawString(MARGIN + c.stringWidth(r.genre, "Helvetica-Bold", 11), y, total_pct)
        y -= 5 * mm
        for sig in r.trope_signals[:4]:
            y = ensure_space(y, 9 * mm)
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(C_MUTED)
            c.drawString(MARGIN, y, sig.trope)
            y -= 4 * mm
            wrapped = simpleSplit('"' + sig.signal + '"',
                                  "Helvetica-Oblique", 8, CONTENT_W - 6 * mm)
            for ln in wrapped:
                y = ensure_space(y, 4 * mm)
                c.setFont("Helvetica-Oblique", 8)
                c.setFillColor(C_TEXT)
                c.drawString(MARGIN + 3 * mm, y, ln)
                y -= 3.5 * mm
            y -= 1.5 * mm
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.3)
        c.line(MARGIN, y, W - MARGIN, y)
        y -= 5 * mm

    c.save()


def _placeholder_entry(parent, placeholder: str, **kwargs) -> tk.Entry:
    """Entry widget with gray placeholder text that clears on focus."""
    entry = tk.Entry(parent, fg=TEXT_MUTED, **kwargs)
    entry.insert(0, placeholder)
    entry._has_placeholder = True

    def _focus_in(_e):
        if entry._has_placeholder:
            entry.delete(0, tk.END)
            entry.config(fg=TEXT)
            entry._has_placeholder = False

    def _focus_out(_e):
        if not entry.get().strip():
            entry.insert(0, placeholder)
            entry.config(fg=TEXT_MUTED)
            entry._has_placeholder = True

    entry.bind("<FocusIn>", _focus_in)
    entry.bind("<FocusOut>", _focus_out)
    return entry


def _placeholder_text(parent, placeholder: str, **kwargs) -> tk.Text:
    """Text widget with gray placeholder text that clears on focus."""
    widget = tk.Text(parent, fg=TEXT_MUTED, **kwargs)
    widget.insert("1.0", placeholder)
    widget._has_placeholder = True

    def _focus_in(_e):
        if widget._has_placeholder:
            widget.delete("1.0", tk.END)
            widget.config(fg=TEXT)
            widget._has_placeholder = False

    def _focus_out(_e):
        if not widget.get("1.0", tk.END).strip():
            widget.insert("1.0", placeholder)
            widget.config(fg=TEXT_MUTED)
            widget._has_placeholder = True

    widget.bind("<FocusIn>", _focus_in)
    widget.bind("<FocusOut>", _focus_out)
    return widget


def _cog_btn(parent, command):
    btn = tk.Button(
        parent, text="\u2699", command=command,
        bg=BG, fg=TEXT_MUTED, font=FONT_COG,
        relief=tk.FLAT, bd=0, padx=6, pady=4,
        cursor="hand2", activebackground=BTN_SEC_H, activeforeground=TEXT,
        highlightthickness=0,
    )
    btn.bind("<Enter>", lambda _: btn.config(fg=ACCENT, bg=BTN_SEC_H))
    btn.bind("<Leave>", lambda _: btn.config(fg=TEXT_MUTED, bg=BG))
    return btn


def _write_opening_report(path: str, title: str, result: OpeningResult) -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.lib.utils import simpleSplit

    C_BG     = HexColor("#fdf9f6")
    C_ACCENT = HexColor("#c9a96e")
    C_ACCENT2= HexColor("#b8954e")
    C_TEXT   = HexColor("#111111")
    C_MUTED  = HexColor("#6b7280")
    C_BORDER = HexColor("#d4ba90")
    C_HDR    = HexColor("#ede0c8")
    C_ROW0   = HexColor("#fdf9f6")
    C_ROW1   = HexColor("#ffffff")
    C_WHITE  = white

    W, H    = A4
    MARGIN  = 20 * mm
    CONT_W  = W - 2 * MARGIN
    BAND_H  = 18 * mm
    FOOT_H  = 10 * mm
    TOP_Y   = H - BAND_H - 6 * mm

    COL1_W = 44 * mm   # Checklist item
    COL3_W = 28 * mm   # Confidence
    COL2_W = CONT_W - COL1_W - COL3_W

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm
    page_num  = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror -- {title} -- Opening Analysis")

    def _draw_chrome():
        page_num[0] += 1
        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2); c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)
        mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, mid_y, "Opening Analysis")
        try:
            c.drawImage(logo_path, W - MARGIN - logo_size,
                        H - BAND_H + (BAND_H - logo_size) / 2,
                        width=logo_size, height=logo_size,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
        c.line(MARGIN, FOOT_H, W - MARGIN, FOOT_H)
        c.setFont("Helvetica", 8); c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOT_H - 4 * mm, f"Page {page_num[0]}")

    def new_page():
        c.showPage(); _draw_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOT_H + 4 * mm:
            new_page(); return TOP_Y
        return y

    _draw_chrome()
    y = TOP_Y

    c.setFont("Helvetica", 10); c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 8 * mm

    # header row
    HDR_H = 8 * mm
    c.setFillColor(C_HDR)
    c.rect(MARGIN, y - HDR_H, CONT_W, HDR_H, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
    hdr_y = y - HDR_H + 2.5 * mm
    c.drawString(MARGIN + 3 * mm, hdr_y, "Checklist")
    c.drawString(MARGIN + COL1_W + 3 * mm, hdr_y, "Reader Guess")
    c.drawString(MARGIN + COL1_W + COL2_W + 3 * mm, hdr_y, "Reader Confidence")
    y -= HDR_H


    for row_idx, item in enumerate(result.items):
        guess_lines = simpleSplit(item.guess, "Helvetica", 9, COL2_W - 6 * mm)
        row_h = max(8 * mm, len(guess_lines) * 4.8 * mm + 5 * mm)
        y = ensure_space(y, row_h)
        bg = C_ROW0 if row_idx % 2 == 0 else C_ROW1
        c.setFillColor(bg)
        c.rect(MARGIN, y - row_h, CONT_W, row_h, fill=1, stroke=0)

        text_y = y - 4 * mm
        c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
        c.drawString(MARGIN + 3 * mm, text_y, item.item)

        c.setFont("Helvetica", 9)
        for ln in guess_lines:
            c.drawString(MARGIN + COL1_W + 3 * mm, text_y, ln)
            text_y -= 4.8 * mm

        emoji_str = CONFIDENCE_EMOJI.get(item.confidence, "\U0001f610")
        c.setFont("Helvetica", 9); c.setFillColor(C_MUTED)
        c.drawString(MARGIN + COL1_W + COL2_W + 3 * mm, y - 4 * mm,
                     f"{emoji_str}  {item.confidence}")
        y -= row_h

    # bottom border
    c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)

    c.save()


def _write_plot_report(path: str, title: str, synopsis_result: "SynopsisResult",
                       arcs_result: "ArcsResult | None") -> None:
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas
    from reportlab.lib.utils import simpleSplit

    C_BG     = HexColor("#fdf9f6")
    C_ACCENT = HexColor("#c9a96e")
    C_ACCENT2= HexColor("#b8954e")
    C_TEXT   = HexColor("#111111")
    C_MUTED  = HexColor("#6b7280")
    C_BORDER = HexColor("#d4ba90")
    C_HDR    = HexColor("#ede0c8")
    C_WHITE  = white

    W, H    = A4
    MARGIN  = 20 * mm
    CONT_W  = W - 2 * MARGIN
    BAND_H  = 18 * mm
    FOOT_H  = 10 * mm
    TOP_Y   = H - BAND_H - 6 * mm

    logo_path = str(ICON_PATH)
    logo_size = 13 * mm
    page_num  = [0]

    c = PDFCanvas(path, pagesize=A4)
    c.setTitle(f"Mirror -- {title} -- Plot Analysis")

    def _draw_chrome():
        page_num[0] += 1
        c.setFillColor(C_BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(C_ACCENT)
        c.rect(0, H - BAND_H, W, BAND_H, fill=1, stroke=0)
        c.setStrokeColor(C_ACCENT2); c.setLineWidth(1)
        c.line(0, H - BAND_H, W, H - BAND_H)
        mid_y = H - BAND_H + (BAND_H - 5 * mm) / 2
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_WHITE)
        c.drawString(MARGIN, mid_y, "Plot Analysis")
        try:
            c.drawImage(logo_path, W - MARGIN - logo_size,
                        H - BAND_H + (BAND_H - logo_size) / 2,
                        width=logo_size, height=logo_size,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
        c.line(MARGIN, FOOT_H, W - MARGIN, FOOT_H)
        c.setFont("Helvetica", 8); c.setFillColor(C_MUTED)
        c.drawCentredString(W / 2, FOOT_H - 4 * mm, f"Page {page_num[0]}")

    def new_page():
        c.showPage(); _draw_chrome()

    def ensure_space(y, needed):
        if y - needed < FOOT_H + 4 * mm:
            new_page(); return TOP_Y
        return y

    def section_header(y, label):
        y = ensure_space(y, 12 * mm)
        c.setFillColor(C_HDR)
        c.rect(MARGIN, y - 7 * mm, CONT_W, 7 * mm, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
        c.drawString(MARGIN + 3 * mm, y - 5 * mm, label)
        return y - 7 * mm

    _draw_chrome()
    y = TOP_Y

    c.setFont("Helvetica", 10); c.setFillColor(C_MUTED)
    c.drawRightString(W - MARGIN, y, title)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    c.drawRightString(W - MARGIN, y, date.today().strftime("%d %B %Y"))
    y -= 3 * mm
    c.setStrokeColor(C_BORDER); c.setLineWidth(0.5)
    c.line(MARGIN, y, W - MARGIN, y)
    y -= 8 * mm

    # logline
    if synopsis_result.logline:
        log_lines = simpleSplit(synopsis_result.logline, "Helvetica-Oblique", 11, CONT_W)
        y = ensure_space(y, len(log_lines) * 6 * mm + 4 * mm)
        c.setFont("Helvetica-Oblique", 11); c.setFillColor(C_ACCENT)
        for ln in log_lines:
            c.drawString(MARGIN, y, ln)
            y -= 6 * mm
        y -= 4 * mm

    # synopsis
    if synopsis_result.synopsis:
        syn_lines = simpleSplit(synopsis_result.synopsis, "Helvetica", 9, CONT_W)
        for ln in syn_lines:
            y = ensure_space(y, 5 * mm)
            c.setFont("Helvetica", 9); c.setFillColor(C_TEXT)
            c.drawString(MARGIN, y, ln)
            y -= 5 * mm
        y -= 6 * mm

    # beats
    if synopsis_result.plot_beats:
        y = section_header(y, "BEATS")
        y -= 4 * mm
        for beat in synopsis_result.plot_beats:
            y = ensure_space(y, 10 * mm)
            c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
            c.drawString(MARGIN, y, beat.term)
            sum_lines = simpleSplit(beat.summary, "Helvetica", 9, CONT_W - 4 * mm)
            y -= 5 * mm
            for ln in sum_lines:
                y = ensure_space(y, 5 * mm)
                c.setFont("Helvetica", 9); c.setFillColor(C_MUTED)
                c.drawString(MARGIN + 4 * mm, y, ln)
                y -= 5 * mm
            y -= 3 * mm

    # arcs
    if arcs_result:
        if arcs_result.structural:
            y -= 4 * mm
            y = section_header(y, "STRUCTURAL ARCS")
            y -= 4 * mm
            for arc in arcs_result.structural:
                y = ensure_space(y, 10 * mm)
                c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
                c.drawString(MARGIN, y, arc.name)
                y -= 5 * mm
                c.setFont("Helvetica", 9); c.setFillColor(C_MUTED)
                c.drawString(MARGIN + 4 * mm, y, f"{arc.start}  →  {arc.end}")
                y -= 6 * mm

        if arcs_result.character:
            y -= 4 * mm
            y = section_header(y, "CHARACTER ARCS")
            y -= 4 * mm
            for arc in arcs_result.character:
                y = ensure_space(y, 10 * mm)
                c.setFont("Helvetica-Bold", 9); c.setFillColor(C_TEXT)
                c.drawString(MARGIN, y, arc.name)
                y -= 5 * mm
                c.setFont("Helvetica", 9); c.setFillColor(C_MUTED)
                c.drawString(MARGIN + 4 * mm, y, f"{arc.start}  →  {arc.end}")
                y -= 6 * mm

    c.save()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Mirror")
        self.geometry("860x680")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.after(0, lambda: self.state("zoomed"))

        self._filepath: str | None = None
        self._results:  list[ChunkResult] = []
        self._sheets:   list[CharacterSheet] = []
        self._emotion_sheets: list[CharacterSheet] = []
        self._logo_img_large = None

        self._model_var_opening = tk.StringVar(value="Sonnet (best)")
        self._model_var_synopsis = tk.StringVar(value=next(iter(MODELS)))
        self._model_var_copy  = tk.StringVar(value=next(iter(MODELS)))
        self._model_var_arch  = tk.StringVar(value=next(iter(MODELS)))
        self._model_var_emot  = tk.StringVar(value=next(iter(MODELS)))
        self._model_var_pers  = tk.StringVar(value=next(iter(MODELS)))
        self._model_var_genre = tk.StringVar(value=next(iter(MODELS)))
        self._synopsis_result: SynopsisResult | None = None
        self._arcs_result:    ArcsResult | None = None
        self._opening_result: OpeningResult | None = None
        self._focal_chars_var      = tk.StringVar(value="")
        self._focal_chars_emot_var = tk.StringVar(value="")
        self._focal_chars_pers_var = tk.StringVar(value="")
        self._enabled_vars:    dict[str, tk.BooleanVar] = {}
        self._archetype_vars:  dict[str, tk.BooleanVar] = {}
        self._emotion_vars:    dict[str, tk.BooleanVar] = {}
        self._personality_vars: dict[str, tk.BooleanVar] = {}
        self._genre_vars:      dict[str, tk.BooleanVar] = {}
        self._opening_item_vars: dict[str, tk.BooleanVar] = {}
        self._arc_structural_vars: dict[str, tk.BooleanVar] = {}
        self._arc_character_var = tk.BooleanVar(value=True)
        self._style_vars:     dict[str, tk.StringVar]  = {}

        self._custom_archetypes:  list[dict] = []
        self._custom_emotions:    list[dict] = []
        self._custom_personalities: list[dict] = []
        self._custom_genres:      list[dict] = []

        self._arch_grid:       tk.Frame | None = None
        self._emot_grid:       tk.Frame | None = None
        self._pers_grid:       tk.Frame | None = None
        self._genre_grid:      tk.Frame | None = None
        self._arch_grid_count  = 0
        self._emot_grid_count  = 0
        self._pers_grid_count  = 0
        self._genre_grid_count = 0

        self._personality_sheets: list[CharacterSheet] = []

        self._emot_group_by = tk.StringVar(value="emotion")
        self._pers_group_by = tk.StringVar(value="trait")

        self._copy_settings_open    = False
        self._arch_settings_open    = False
        self._emot_settings_open    = False
        self._pers_settings_open    = False
        self._opening_settings_open = False
        self._syn_settings_open     = False

        self._build()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _unique_id(self, name: str, existing: dict) -> str:
        base = re.sub(r"[^\w]+", "_", name.lower()).strip("_") or "custom"
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    # ── build ──────────────────────────────────────────────────────────────────

    def _build(self):
        self._load_icon()
        self._apply_ttk_theme()

        # ── landing (cream, logo blends in) ───────────────
        self._landing = tk.Frame(self, bg=BG)
        self._landing.pack(fill=tk.BOTH, expand=True)

        center = tk.Frame(self._landing, bg=BG)
        center.place(relx=0.5, rely=0.44, anchor="center")

        if self._logo_img_large:
            tk.Label(center, image=self._logo_img_large, bg=BG).pack()

        tk.Frame(center, height=28, bg=BG).pack()

        # dark-on-cream open button
        open_btn = tk.Button(
            center, text="Open manuscript…", command=self._pick_file,
            bg=BG, fg=TEXT, font=FONT_UI,
            relief=tk.FLAT, bd=0, padx=14, pady=7,
            cursor="hand2", activebackground=BG, activeforeground=TEXT,
            highlightthickness=0,
        )
        open_btn.bind("<Enter>", lambda _: open_btn.config(font=("Georgia", 11, "underline")))
        open_btn.bind("<Leave>", lambda _: open_btn.config(font=FONT_UI))
        open_btn.pack()

        # ── post-open (dark, revealed after file chosen) ──
        self._post_open = tk.Frame(self, bg=BG)

        file_bar = tk.Frame(self._post_open, bg=BG, padx=18, pady=9)
        file_bar.pack(fill=tk.X)

        self._file_lbl = tk.Label(
            file_bar, text="", font=FONT_TITLE, fg=TEXT, bg=BG, anchor="w"
        )
        self._file_lbl.pack(side=tk.LEFT)

        btn_change = tk.Button(
            file_bar, text="Change file", command=self._pick_file,
            font=FONT_SMALL, fg=TEXT_MUTED, bg=BG,
            relief=tk.FLAT, bd=0, cursor="hand2",
            activeforeground=TEXT, activebackground=BG,
            highlightthickness=0,
        )
        btn_change.pack(side=tk.RIGHT, padx=(0, 2))
        btn_change.bind("<Enter>", lambda _: btn_change.config(fg=TEXT))
        btn_change.bind("<Leave>", lambda _: btn_change.config(fg=TEXT_MUTED))

        btn_del = tk.Button(
            file_bar, text="Delete caches", command=self._delete_caches,
            font=FONT_SMALL, fg=TEXT_MUTED, bg=BG,
            relief=tk.FLAT, bd=0, cursor="hand2",
            activeforeground=TEXT, activebackground=BG,
            highlightthickness=0,
        )
        btn_del.pack(side=tk.RIGHT, padx=(0, 12))
        btn_del.bind("<Enter>", lambda _: btn_del.config(fg=TEXT))
        btn_del.bind("<Leave>", lambda _: btn_del.config(fg=TEXT_MUTED))

        tk.Frame(self._post_open, bg=BORDER, height=1).pack(fill=tk.X)

        # notebook
        self._notebook = ttk.Notebook(self._post_open, style="Mirror.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._build_opening_tab()
        self._build_synopsis_tab()
        self._build_archetypes_tab()
        self._build_emotions_tab()
        self._build_personality_tab()
        self._build_genre_tab()
        self._build_copyedit_tab()

    def _apply_ttk_theme(self):
        style = ttk.Style(self)
        style.configure("Mirror.TNotebook",
                         background=BG, borderwidth=0)
        style.configure("Mirror.TNotebook.Tab",
                         background=BTN_SEC, foreground=TEXT_MUTED,
                         font=FONT_UI, padding=(16, 5))
        style.map("Mirror.TNotebook.Tab",
                  background=[("selected", BG), ("active", BTN_SEC_H)],
                  foreground=[("selected", ACCENT), ("active", TEXT)])
        # combobox field colors
        style.configure("TCombobox",
                         fieldbackground=BG_SETTINGS,
                         background=BTN_SEC,
                         foreground=TEXT,
                         arrowcolor=TEXT_MUTED,
                         selectbackground=BG_SETTINGS,
                         selectforeground=TEXT)

    def _build_opening_tab(self):
        open_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(open_tab, text="Opening")

        open_bar = tk.Frame(open_tab, bg=BG, padx=18, pady=8)
        open_bar.pack(fill=tk.X)

        self._btn_analyze_opening = _flat_btn(
            open_bar, "Analyze", self._start_opening,
            primary=True, state=tk.DISABLED,
        )
        self._btn_analyze_opening.pack(side=tk.LEFT)

        self._status_opening = tk.Label(
            open_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_opening.pack(side=tk.LEFT, padx=(12, 0))

        self._btn_save_opening = _flat_btn(
            open_bar, "Save PDF report", self._save_opening_report
        )

        self._cog_opening = _cog_btn(open_bar, self._toggle_opening_settings)
        self._cog_opening.pack(side=tk.RIGHT)

        tk.Frame(open_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._opening_settings = self._build_opening_settings(open_tab)

        open_outer = tk.Frame(open_tab, bg=BG, padx=18, pady=14)
        open_outer.pack(fill=tk.BOTH, expand=True)

        self._open_canvas = tk.Canvas(
            open_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        open_scrollbar = tk.Scrollbar(
            open_outer, orient=tk.VERTICAL, command=self._open_canvas.yview
        )
        self._open_canvas.configure(yscrollcommand=open_scrollbar.set)
        open_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._open_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._open_frame = tk.Frame(self._open_canvas, bg=BG_RESULTS)
        self._open_window = self._open_canvas.create_window(
            (0, 0), window=self._open_frame, anchor="nw"
        )
        self._open_frame.bind(
            "<Configure>",
            lambda _: self._open_canvas.configure(
                scrollregion=self._open_canvas.bbox("all")
            ),
        )
        self._open_canvas.bind(
            "<Configure>",
            lambda e: self._open_canvas.itemconfig(self._open_window, width=e.width),
        )
        self._open_canvas.bind(
            "<MouseWheel>",
            lambda e: self._open_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_opening_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_opening,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 10))

        hdr = tk.Frame(panel, bg=BG_SETTINGS)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="CHECKLIST ITEMS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        for label, val in [("none", False), ("·", None), ("all", True)]:
            if val is None:
                tk.Label(hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                          bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                          activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                          command=lambda v=val: [x.set(v) for x in self._opening_item_vars.values()]
                          ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        grid = tk.Frame(panel, bg=BG_SETTINGS)
        grid.pack(fill=tk.X)
        for idx, item in enumerate(CHECKLIST_ITEMS):
            var = tk.BooleanVar(value=True)
            self._opening_item_vars[item] = var
            col = idx % 2
            row = idx // 2
            cb = tk.Checkbutton(
                grid, text=item, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, selectcolor=BG_SETTINGS,
                anchor="w", cursor="hand2",
            )
            cb.grid(row=row, column=col, sticky="w", padx=(0, 20), pady=1)

        return panel

    def _toggle_opening_settings(self):
        if self._opening_settings_open:
            self._opening_settings.pack_forget()
            self._opening_settings_open = False
        else:
            self._opening_settings.pack(fill=tk.X, after=self._cog_opening.master)
            self._opening_settings_open = True

    def _build_syn_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=10)
        all_arcs = load_structural_arcs()

        row0 = tk.Frame(panel, bg=BG_SETTINGS)
        row0.pack(fill=tk.X, pady=(0, 8))
        tk.Label(row0, text="Model:", font=FONT_SMALL, fg=TEXT, bg=BG_SETTINGS).pack(side=tk.LEFT)
        for name in MODELS:
            tk.Radiobutton(
                row0, text=name, variable=self._model_var_synopsis, value=name,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT, selectcolor=BG_SETTINGS,
                activebackground=BG_SETTINGS,
            ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(panel, text="Structural arcs:", font=FONT_SMALL, fg=TEXT, bg=BG_SETTINGS).pack(
            anchor="w", pady=(4, 4)
        )
        arc_grid = tk.Frame(panel, bg=BG_SETTINGS)
        arc_grid.pack(fill=tk.X)
        cols = 3
        for i, arc in enumerate(all_arcs):
            var = tk.BooleanVar(value=True)
            self._arc_structural_vars[arc["id"]] = var
            tk.Checkbutton(
                arc_grid, text=arc["name"], variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, selectcolor=BG_SETTINGS,
            ).grid(row=i // cols, column=i % cols, sticky="w", padx=(0, 12))

        char_row = tk.Frame(panel, bg=BG_SETTINGS)
        char_row.pack(fill=tk.X, pady=(10, 0))
        tk.Checkbutton(
            char_row, text="Include character arcs", variable=self._arc_character_var,
            font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
            activebackground=BG_SETTINGS, selectcolor=BG_SETTINGS,
        ).pack(side=tk.LEFT)

        return panel

    def _toggle_syn_settings(self):
        if self._syn_settings_open:
            self._syn_settings.pack_forget()
            self._syn_settings_open = False
        else:
            self._syn_settings.pack(fill=tk.X, after=self._cog_syn.master)
            self._syn_settings_open = True

    def _build_synopsis_tab(self):
        syn_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(syn_tab, text="Plot")

        syn_bar = tk.Frame(syn_tab, bg=BG, padx=18, pady=8)
        syn_bar.pack(fill=tk.X)

        self._btn_regen_synopsis = _flat_btn(
            syn_bar, "Analyze", self._start_synopsis,
            primary=True, state=tk.DISABLED,
        )
        self._btn_regen_synopsis.pack(side=tk.LEFT)

        self._status_synopsis = tk.Label(
            syn_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_synopsis.pack(side=tk.LEFT, padx=(12, 0))

        self._btn_save_syn = _flat_btn(syn_bar, "Save PDF report", self._save_syn_report)

        self._cog_syn = _cog_btn(syn_bar, self._toggle_syn_settings)
        self._cog_syn.pack(side=tk.RIGHT)

        tk.Frame(syn_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._syn_settings = self._build_syn_settings(syn_tab)

        syn_outer = tk.Frame(syn_tab, bg=BG, padx=18, pady=14)
        syn_outer.pack(fill=tk.BOTH, expand=True)

        self._syn_canvas = tk.Canvas(
            syn_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        syn_scrollbar = tk.Scrollbar(
            syn_outer, orient=tk.VERTICAL, command=self._syn_canvas.yview
        )
        self._syn_canvas.configure(yscrollcommand=syn_scrollbar.set)
        syn_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._syn_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._syn_frame = tk.Frame(self._syn_canvas, bg=BG_RESULTS)
        self._syn_window = self._syn_canvas.create_window(
            (0, 0), window=self._syn_frame, anchor="nw"
        )
        self._syn_frame.bind(
            "<Configure>",
            lambda _: self._syn_canvas.configure(
                scrollregion=self._syn_canvas.bbox("all")
            ),
        )
        self._syn_canvas.bind(
            "<Configure>",
            lambda e: self._syn_canvas.itemconfig(self._syn_window, width=e.width),
        )
        self._syn_canvas.bind(
            "<MouseWheel>",
            lambda e: self._syn_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_copyedit_tab(self):
        copy_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(copy_tab, text="Copyedit")

        copy_bar = tk.Frame(copy_tab, bg=BG, padx=18, pady=8)
        copy_bar.pack(fill=tk.X)

        self._btn_analyze_copy = _flat_btn(
            copy_bar, "Analyze", self._start_copyedit,
            primary=True, state=tk.DISABLED
        )
        self._btn_analyze_copy.pack(side=tk.LEFT)

        self._status_copy = tk.Label(
            copy_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_copy.pack(side=tk.LEFT, padx=(12, 0))

        # save button inserted after status via after= when revealed
        self._btn_save = _flat_btn(copy_bar, "Save reviewed .docx", self._save)

        self._cog_copy = _cog_btn(copy_bar, self._toggle_copy_settings)
        self._cog_copy.pack(side=tk.RIGHT)

        tk.Frame(copy_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._copy_settings = self._build_copy_settings(copy_tab)

        self._results_outer = tk.Frame(copy_tab, bg=BG, padx=18, pady=14)
        self._results_outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            self._results_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        scrollbar = tk.Scrollbar(
            self._results_outer, orient=tk.VERTICAL, command=self._canvas.yview
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._results_frame = tk.Frame(self._canvas, bg=BG_RESULTS)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._results_frame, anchor="nw"
        )
        self._results_frame.bind("<Configure>", self._on_frame_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _build_archetypes_tab(self):
        arch_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(arch_tab, text="Archetypes")

        arch_bar = tk.Frame(arch_tab, bg=BG, padx=18, pady=8)
        arch_bar.pack(fill=tk.X)

        self._btn_analyze_arch = _flat_btn(
            arch_bar, "Analyze", self._start_archetypes,
            primary=True, state=tk.DISABLED
        )
        self._btn_analyze_arch.pack(side=tk.LEFT)

        self._status_arch = tk.Label(
            arch_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_arch.pack(side=tk.LEFT, padx=(12, 0))

        # save button inserted after status via after= when revealed
        self._btn_save_arch = _flat_btn(arch_bar, "Save PDF report", self._save_arch_report)

        self._cog_arch = _cog_btn(arch_bar, self._toggle_arch_settings)
        self._cog_arch.pack(side=tk.RIGHT)

        tk.Frame(arch_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._arch_settings = self._build_arch_settings(arch_tab)

        self._char_outer = tk.Frame(arch_tab, bg=BG, padx=18, pady=14)
        self._char_outer.pack(fill=tk.BOTH, expand=True)

        self._char_canvas = tk.Canvas(
            self._char_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        char_scrollbar = tk.Scrollbar(
            self._char_outer, orient=tk.VERTICAL, command=self._char_canvas.yview
        )
        self._char_canvas.configure(yscrollcommand=char_scrollbar.set)
        char_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._char_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._char_frame = tk.Frame(self._char_canvas, bg=BG_RESULTS)
        self._char_window = self._char_canvas.create_window(
            (0, 0), window=self._char_frame, anchor="nw"
        )
        self._char_frame.bind(
            "<Configure>",
            lambda _: self._char_canvas.configure(
                scrollregion=self._char_canvas.bbox("all")
            ),
        )
        self._char_canvas.bind(
            "<Configure>",
            lambda e: self._char_canvas.itemconfig(self._char_window, width=e.width),
        )
        self._char_canvas.bind(
            "<MouseWheel>",
            lambda e: self._char_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_emotions_tab(self):
        emot_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(emot_tab, text="Emotions")

        emot_bar = tk.Frame(emot_tab, bg=BG, padx=18, pady=8)
        emot_bar.pack(fill=tk.X)

        self._btn_analyze_emot = _flat_btn(
            emot_bar, "Analyze", self._start_emotions,
            primary=True, state=tk.DISABLED
        )
        self._btn_analyze_emot.pack(side=tk.LEFT)

        self._status_emot = tk.Label(
            emot_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_emot.pack(side=tk.LEFT, padx=(12, 0))

        self._btn_save_emot = _flat_btn(emot_bar, "Save PDF report", self._save_emotion_report)

        self._cog_emot = _cog_btn(emot_bar, self._toggle_emot_settings)
        self._cog_emot.pack(side=tk.RIGHT)

        tk.Frame(emot_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._emot_settings = self._build_emot_settings(emot_tab)

        self._emot_outer = tk.Frame(emot_tab, bg=BG, padx=18, pady=14)
        self._emot_outer.pack(fill=tk.BOTH, expand=True)

        self._emot_canvas = tk.Canvas(
            self._emot_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        emot_scrollbar = tk.Scrollbar(
            self._emot_outer, orient=tk.VERTICAL, command=self._emot_canvas.yview
        )
        self._emot_canvas.configure(yscrollcommand=emot_scrollbar.set)
        emot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._emot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._emot_frame = tk.Frame(self._emot_canvas, bg=BG_RESULTS)
        self._emot_window = self._emot_canvas.create_window(
            (0, 0), window=self._emot_frame, anchor="nw"
        )
        self._emot_frame.bind(
            "<Configure>",
            lambda _: self._emot_canvas.configure(
                scrollregion=self._emot_canvas.bbox("all")
            ),
        )
        self._emot_canvas.bind(
            "<Configure>",
            lambda e: self._emot_canvas.itemconfig(self._emot_window, width=e.width),
        )
        self._emot_canvas.bind(
            "<MouseWheel>",
            lambda e: self._emot_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_personality_tab(self):
        pers_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(pers_tab, text="Personality")

        pers_bar = tk.Frame(pers_tab, bg=BG, padx=18, pady=8)
        pers_bar.pack(fill=tk.X)

        self._btn_analyze_pers = _flat_btn(
            pers_bar, "Analyze", self._start_personality,
            primary=True, state=tk.DISABLED
        )
        self._btn_analyze_pers.pack(side=tk.LEFT)

        self._status_pers = tk.Label(
            pers_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_pers.pack(side=tk.LEFT, padx=(12, 0))

        self._btn_save_pers = _flat_btn(pers_bar, "Save PDF report", self._save_personality_report)

        self._cog_pers = _cog_btn(pers_bar, self._toggle_pers_settings)
        self._cog_pers.pack(side=tk.RIGHT)

        tk.Frame(pers_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._pers_settings = self._build_pers_settings(pers_tab)

        self._pers_outer = tk.Frame(pers_tab, bg=BG, padx=18, pady=14)
        self._pers_outer.pack(fill=tk.BOTH, expand=True)

        self._pers_canvas = tk.Canvas(
            self._pers_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        pers_scrollbar = tk.Scrollbar(
            self._pers_outer, orient=tk.VERTICAL, command=self._pers_canvas.yview
        )
        self._pers_canvas.configure(yscrollcommand=pers_scrollbar.set)
        pers_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._pers_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._pers_frame = tk.Frame(self._pers_canvas, bg=BG_RESULTS)
        self._pers_window = self._pers_canvas.create_window(
            (0, 0), window=self._pers_frame, anchor="nw"
        )
        self._pers_frame.bind(
            "<Configure>",
            lambda _: self._pers_canvas.configure(
                scrollregion=self._pers_canvas.bbox("all")
            ),
        )
        self._pers_canvas.bind(
            "<Configure>",
            lambda e: self._pers_canvas.itemconfig(self._pers_window, width=e.width),
        )
        self._pers_canvas.bind(
            "<MouseWheel>",
            lambda e: self._pers_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_genre_tab(self):
        genre_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(genre_tab, text="Genres")

        genre_bar = tk.Frame(genre_tab, bg=BG, padx=18, pady=8)
        genre_bar.pack(fill=tk.X)

        self._btn_analyze_genre = _flat_btn(
            genre_bar, "Analyze", self._start_genre,
            primary=True, state=tk.DISABLED,
        )
        self._btn_analyze_genre.pack(side=tk.LEFT)

        self._status_genre = tk.Label(
            genre_bar, text="", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG
        )
        self._status_genre.pack(side=tk.LEFT, padx=(12, 0))

        self._btn_save_genre = _flat_btn(genre_bar, "Save PDF report", self._save_genre_report)

        self._cog_genre = _cog_btn(genre_bar, self._toggle_genre_settings)
        self._cog_genre.pack(side=tk.RIGHT)

        tk.Frame(genre_tab, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self._genre_settings = self._build_genre_settings(genre_tab)
        self._genre_settings_open = False

        self._genre_outer = tk.Frame(genre_tab, bg=BG, padx=18, pady=14)
        self._genre_outer.pack(fill=tk.BOTH, expand=True)

        self._genre_canvas = tk.Canvas(
            self._genre_outer, bg=BG_RESULTS, highlightthickness=0, bd=0
        )
        genre_scrollbar = tk.Scrollbar(
            self._genre_outer, orient=tk.VERTICAL, command=self._genre_canvas.yview
        )
        self._genre_canvas.configure(yscrollcommand=genre_scrollbar.set)
        genre_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._genre_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._genre_frame = tk.Frame(self._genre_canvas, bg=BG_RESULTS)
        self._genre_window = self._genre_canvas.create_window(
            (0, 0), window=self._genre_frame, anchor="nw"
        )
        self._genre_frame.bind(
            "<Configure>",
            lambda _: self._genre_canvas.configure(
                scrollregion=self._genre_canvas.bbox("all")
            ),
        )
        self._genre_canvas.bind(
            "<Configure>",
            lambda e: self._genre_canvas.itemconfig(self._genre_window, width=e.width),
        )
        self._genre_canvas.bind(
            "<MouseWheel>",
            lambda e: self._genre_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )
        self._genre_results: list[GenreResult] = []

    # ── settings panels ────────────────────────────────────────────────────────

    def _build_copy_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_copy,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 10))

        left = tk.Frame(panel, bg=BG_SETTINGS)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor="n")

        hdr = tk.Frame(left, bg=BG_SETTINGS)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="ERROR TYPES", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        for label, val in [("none", False), ("·", None), ("all", True)]:
            if val is None:
                tk.Label(hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                          bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                          activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                          command=lambda v=val: [x.set(v) for x in self._enabled_vars.values()]
                          ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        grid = tk.Frame(left, bg=BG_SETTINGS)
        grid.pack(fill=tk.X)
        for idx, entry in enumerate(load_error_types()):
            etype = entry["type"]
            var = tk.BooleanVar(value=True)
            self._enabled_vars[etype] = var
            row, col = divmod(idx, 3)
            tk.Checkbutton(
                grid, text=etype, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS,
                anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 12), pady=1)

        tk.Frame(panel, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=16)

        right = tk.Frame(panel, bg=BG_SETTINGS)
        right.pack(side=tk.LEFT, anchor="n")
        tk.Label(right, text="HOUSE STYLE", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        style_cfg = load_style_config()
        for row_idx, (key, label) in enumerate(STYLE_LABELS.items(), start=1):
            cfg     = style_cfg.get(key, {})
            options = list(cfg.get("rules", {}).keys())
            current = cfg.get("default", options[0] if options else "")
            var = tk.StringVar(value=current)
            self._style_vars[key] = var
            tk.Label(right, text=label + ":", font=FONT_SMALL,
                     fg=TEXT, bg=BG_SETTINGS, anchor="w"
                     ).grid(row=row_idx, column=0, sticky="w", pady=3, padx=(0, 8))
            cb = ttk.Combobox(right, textvariable=var, values=options,
                              state="readonly", width=14, font=FONT_SMALL)
            cb.grid(row=row_idx, column=1, sticky="w", pady=3)
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self._on_style_change(k))

        return panel

    # ── custom-item restore (from cache) ──────────────────────────────────────

    def _infer_custom_from_cache(self, kind: str, cached: dict) -> list:
        """Derive custom items from cached sheet/result data when they weren't stored explicitly."""
        if kind == "archetype":
            known = {a["id"] for a in load_archetypes()}
            found: dict = {}
            for sheet in cached.get("sheets", []):
                for arch_id, arch_name, *_ in sheet.top_archetypes:
                    if arch_id not in known and arch_id not in found:
                        found[arch_id] = {"id": arch_id, "name": arch_name, "description": ""}
            return list(found.values())
        elif kind == "emotion":
            known = {e["id"] for e in load_emotions()}
            found = {}
            for sheet in cached.get("sheets", []):
                for eid, ename, *_ in sheet.top_archetypes:
                    if eid not in known and eid not in found:
                        found[eid] = {"id": eid, "name": ename, "description": ""}
            return list(found.values())
        elif kind == "trait":
            known = {t["id"] for t in load_personality_traits()}
            found = {}
            for sheet in cached.get("sheets", []):
                for tid, tname, *_ in sheet.top_archetypes:
                    if tid not in known and tid not in found:
                        found[tid] = {"id": tid, "name": tname, "description": ""}
            return list(found.values())
        elif kind == "genre":
            known = {g["name"] for g in load_genres()}
            found = {}
            for result in cached.get("results", []):
                name = result.genre if hasattr(result, "genre") else result.get("genre", "")
                if name and name not in known and name not in found:
                    found[name] = {"name": name, "description": "", "tropes": []}
            return list(found.values())
        return []

    def _restore_custom_items(self, kind: str, items: list) -> None:
        """Re-insert cached custom elements into state and settings grid (main-thread only)."""
        if not items:
            return
        if kind == "archetype":
            for item in items:
                aid = item.get("id", "")
                if not aid or aid in self._archetype_vars:
                    continue
                self._custom_archetypes.append(item)
                var = tk.BooleanVar(value=True)
                self._archetype_vars[aid] = var
                cols = 5
                row, col = divmod(self._arch_grid_count, cols)
                tk.Checkbutton(
                    self._arch_grid, text=item["name"], variable=var,
                    font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                    activebackground=BG_SETTINGS, activeforeground=TEXT,
                    selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
                ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
                self._arch_grid_count += 1
        elif kind == "emotion":
            for item in items:
                eid = item.get("id", "")
                if not eid or eid in self._emotion_vars:
                    continue
                self._custom_emotions.append(item)
                var = tk.BooleanVar(value=True)
                self._emotion_vars[eid] = var
                cols = 5
                row, col = divmod(self._emot_grid_count, cols)
                tk.Checkbutton(
                    self._emot_grid, text=item["name"], variable=var,
                    font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                    activebackground=BG_SETTINGS, activeforeground=TEXT,
                    selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
                ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
                self._emot_grid_count += 1
        elif kind == "trait":
            for item in items:
                tid = item.get("id", "")
                if not tid or tid in self._personality_vars:
                    continue
                self._custom_personalities.append(item)
                var = tk.BooleanVar(value=True)
                self._personality_vars[tid] = var
                cols = 5
                row, col = divmod(self._pers_grid_count, cols)
                tk.Checkbutton(
                    self._pers_grid, text=item["name"], variable=var,
                    font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                    activebackground=BG_SETTINGS, activeforeground=TEXT,
                    selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
                ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
                self._pers_grid_count += 1
        elif kind == "genre":
            for item in items:
                name = item.get("name", "")
                if not name or name in self._genre_vars:
                    continue
                self._custom_genres.append(item)
                var = tk.BooleanVar(value=True)
                self._genre_vars[name] = var
                cols = 4
                row, col = divmod(self._genre_grid_count, cols)
                tk.Checkbutton(
                    self._genre_grid, text=name, variable=var,
                    font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                    activebackground=BG_SETTINGS, activeforeground=TEXT,
                    selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
                ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
                self._genre_grid_count += 1

    # ── custom-item dialogs ────────────────────────────────────────────────────

    def _open_add_archetype_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add custom archetype")
        win.configure(bg=BG_SETTINGS)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = {"padx": 14, "pady": 6}
        tk.Label(win, text="Name", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=0, column=0, sticky="w", **pad)
        name_entry = _placeholder_entry(
            win, "e.g., The Shapeshifter",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        name_entry.grid(row=0, column=1, sticky="ew", **pad)

        tk.Label(win, text="Description", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=1, column=0, sticky="nw", **pad)
        desc_text = _placeholder_text(
            win,
            "e.g., Adapts identity to suit their audience — reads rooms instantly"
            " and lies fluently, but may not know who they really are.",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=3, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        desc_text.grid(row=1, column=1, sticky="ew", **pad)

        err_lbl = tk.Label(win, text="", font=FONT_SMALL, fg="#c0392b", bg=BG_SETTINGS)
        err_lbl.grid(row=2, column=0, columnspan=2, sticky="w", padx=14)

        btn_row = tk.Frame(win, bg=BG_SETTINGS)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e", padx=14, pady=(4, 10))

        def _add():
            name = ("" if name_entry._has_placeholder else name_entry.get()).strip()
            if not name:
                err_lbl.config(text="Name is required.")
                return
            aid = self._unique_id(name, self._archetype_vars)
            desc = "" if desc_text._has_placeholder else desc_text.get("1.0", tk.END).strip()
            item = {"id": aid, "name": name, "description": desc}
            self._custom_archetypes.append(item)
            var = tk.BooleanVar(value=True)
            self._archetype_vars[aid] = var
            cols = 5
            row, col = divmod(self._arch_grid_count, cols)
            tk.Checkbutton(
                self._arch_grid, text=name, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
            self._arch_grid_count += 1
            win.destroy()

        _flat_btn(btn_row, "Cancel", win.destroy).pack(side=tk.LEFT, padx=(0, 6))
        _flat_btn(btn_row, "Add", _add, primary=True).pack(side=tk.LEFT)
        win.focus_set()
        name_entry.bind("<Return>", lambda _: _add())
        win.wait_window()

    def _open_add_emotion_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add custom emotion")
        win.configure(bg=BG_SETTINGS)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = {"padx": 14, "pady": 6}
        tk.Label(win, text="Name", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=0, column=0, sticky="w", **pad)
        name_entry = _placeholder_entry(
            win, "e.g., Shame",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        name_entry.grid(row=0, column=1, sticky="ew", **pad)

        tk.Label(win, text="Description", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=1, column=0, sticky="nw", **pad)
        desc_text = _placeholder_text(
            win,
            "e.g., Acute self-consciousness triggered by perceived failure or"
            " exposure — the urge to hide, shrink, or disappear.",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=3, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        desc_text.grid(row=1, column=1, sticky="ew", **pad)

        err_lbl = tk.Label(win, text="", font=FONT_SMALL, fg="#c0392b", bg=BG_SETTINGS)
        err_lbl.grid(row=2, column=0, columnspan=2, sticky="w", padx=14)

        btn_row = tk.Frame(win, bg=BG_SETTINGS)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e", padx=14, pady=(4, 10))

        def _add():
            name = ("" if name_entry._has_placeholder else name_entry.get()).strip()
            if not name:
                err_lbl.config(text="Name is required.")
                return
            eid = self._unique_id(name, self._emotion_vars)
            desc = "" if desc_text._has_placeholder else desc_text.get("1.0", tk.END).strip()
            item = {"id": eid, "name": name, "description": desc}
            self._custom_emotions.append(item)
            var = tk.BooleanVar(value=True)
            self._emotion_vars[eid] = var
            cols = 5
            row, col = divmod(self._emot_grid_count, cols)
            tk.Checkbutton(
                self._emot_grid, text=name, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
            self._emot_grid_count += 1
            win.destroy()

        _flat_btn(btn_row, "Cancel", win.destroy).pack(side=tk.LEFT, padx=(0, 6))
        _flat_btn(btn_row, "Add", _add, primary=True).pack(side=tk.LEFT)
        win.focus_set()
        name_entry.bind("<Return>", lambda _: _add())
        win.wait_window()

    def _open_add_personality_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add custom trait")
        win.configure(bg=BG_SETTINGS)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = {"padx": 14, "pady": 6}

        tk.Label(win, text="Dimension name", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=0, column=0, sticky="w", **pad)
        name_entry = _placeholder_entry(
            win, "e.g., Dominance",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        name_entry.grid(row=0, column=1, sticky="ew", **pad)

        tk.Label(win, text="High pole label", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=1, column=0, sticky="w", **pad)
        high_entry = _placeholder_entry(
            win, "e.g., Dominant  (+3)",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        high_entry.grid(row=1, column=1, sticky="ew", **pad)

        tk.Label(win, text="Low pole label", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=2, column=0, sticky="w", **pad)
        low_entry = _placeholder_entry(
            win, "e.g., Submissive  (-3)",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        low_entry.grid(row=2, column=1, sticky="ew", **pad)

        tk.Label(win, text="High pole (+)", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=3, column=0, sticky="nw", **pad)
        high_desc = _placeholder_text(
            win,
            "Describe what scores near +3 look like — assertive, commanding, directs others.",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=2, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        high_desc.grid(row=3, column=1, sticky="ew", **pad)

        tk.Label(win, text="Low pole (−)", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=4, column=0, sticky="nw", **pad)
        low_desc = _placeholder_text(
            win,
            "Describe what scores near -3 look like — deferential, avoids conflict, follows.",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=2, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        low_desc.grid(row=4, column=1, sticky="ew", **pad)

        err_lbl = tk.Label(win, text="", font=FONT_SMALL, fg="#c0392b", bg=BG_SETTINGS)
        err_lbl.grid(row=5, column=0, columnspan=2, sticky="w", padx=14)

        btn_row = tk.Frame(win, bg=BG_SETTINGS)
        btn_row.grid(row=6, column=0, columnspan=2, sticky="e", padx=14, pady=(4, 10))

        def _add():
            name = ("" if name_entry._has_placeholder else name_entry.get()).strip()
            if not name:
                err_lbl.config(text="Dimension name is required.")
                return
            high_n = ("" if high_entry._has_placeholder else high_entry.get()).strip() or name
            low_n  = ("" if low_entry._has_placeholder  else low_entry.get()).strip()  or name
            h_desc = ("" if high_desc._has_placeholder  else high_desc.get("1.0", tk.END).strip())
            l_desc = ("" if low_desc._has_placeholder   else low_desc.get("1.0", tk.END).strip())
            tid  = self._unique_id(name, self._personality_vars)
            item = {
                "id":              tid,
                "name":            name,
                "high_name":       high_n,
                "low_name":        low_n,
                "description":     h_desc,
                "low_description": l_desc,
            }
            self._custom_personalities.append(item)
            var = tk.BooleanVar(value=True)
            self._personality_vars[tid] = var
            cols = 5
            row, col = divmod(self._pers_grid_count, cols)
            tk.Checkbutton(
                self._pers_grid, text=name, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
            self._pers_grid_count += 1
            win.destroy()

        _flat_btn(btn_row, "Cancel", win.destroy).pack(side=tk.LEFT, padx=(0, 6))
        _flat_btn(btn_row, "Add", _add, primary=True).pack(side=tk.LEFT)
        win.focus_set()
        name_entry.bind("<Return>", lambda _: _add())
        win.wait_window()

    def _open_add_genre_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add custom genre")
        win.configure(bg=BG_SETTINGS)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = {"padx": 14, "pady": 6}
        tk.Label(win, text="Name", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=0, column=0, sticky="w", **pad)
        name_entry = _placeholder_entry(
            win, "e.g., Cli-Fi",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        name_entry.grid(row=0, column=1, sticky="ew", **pad)

        tk.Label(win, text="Tropes", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=1, column=0, sticky="nw", **pad)
        tk.Label(win, text="comma-separated\ndrives detection",
                 font=FONT_SMALL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS, justify=tk.LEFT).grid(row=1, column=0, sticky="s", padx=14)
        tropes_text = _placeholder_text(
            win,
            "e.g., dying earth, climate refugee, eco-terrorism, last forest, flooded city",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=3, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        tropes_text.grid(row=1, column=1, sticky="ew", **pad)

        tk.Label(win, text="Description", font=FONT_LABEL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS).grid(row=2, column=0, sticky="nw", **pad)
        tk.Label(win, text="optional,\nnot sent to model",
                 font=FONT_SMALL, fg=TEXT_MUTED,
                 bg=BG_SETTINGS, justify=tk.LEFT).grid(row=2, column=0, sticky="s", padx=14)
        desc_text = _placeholder_text(
            win,
            "e.g., Stories set against climate crisis or ecological collapse.",
            font=FONT_SMALL, bg=BG_RESULTS, relief=tk.FLAT,
            insertbackground=TEXT, width=34, height=3, wrap=tk.WORD,
            highlightthickness=1, highlightbackground=BTN_BORDER,
        )
        desc_text.grid(row=2, column=1, sticky="ew", **pad)

        err_lbl = tk.Label(win, text="", font=FONT_SMALL, fg="#c0392b", bg=BG_SETTINGS)
        err_lbl.grid(row=3, column=0, columnspan=2, sticky="w", padx=14)

        btn_row = tk.Frame(win, bg=BG_SETTINGS)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="e", padx=14, pady=(4, 10))

        def _add():
            name = ("" if name_entry._has_placeholder else name_entry.get()).strip()
            if not name:
                err_lbl.config(text="Name is required.")
                return
            if name in self._genre_vars:
                err_lbl.config(text="A genre with this name already exists.")
                return
            raw_tropes = "" if tropes_text._has_placeholder else tropes_text.get("1.0", tk.END).strip()
            tropes = [t.strip() for t in raw_tropes.split(",") if t.strip()]
            desc = "" if desc_text._has_placeholder else desc_text.get("1.0", tk.END).strip()
            item = {"name": name, "description": desc, "tropes": tropes}
            self._custom_genres.append(item)
            var = tk.BooleanVar(value=True)
            self._genre_vars[name] = var
            cols = 4
            row, col = divmod(self._genre_grid_count, cols)
            tk.Checkbutton(
                self._genre_grid, text=name, variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
            self._genre_grid_count += 1
            win.destroy()

        _flat_btn(btn_row, "Cancel", win.destroy).pack(side=tk.LEFT, padx=(0, 6))
        _flat_btn(btn_row, "Add", _add, primary=True).pack(side=tk.LEFT)
        win.focus_set()
        name_entry.bind("<Return>", lambda _: _add())
        win.wait_window()

    def _build_arch_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        # ── model row ──────────────────────────────────────────
        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X)
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_arch,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        # ── archetypes grid ────────────────────────────────────
        arch_hdr = tk.Frame(panel, bg=BG_SETTINGS)
        arch_hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(arch_hdr, text="ARCHETYPES", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        tk.Button(
            arch_hdr, text="+ add", font=FONT_SMALL, relief=tk.FLAT,
            bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
            activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
            command=self._open_add_archetype_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        for label, val in [("none", False), ("·", None), ("all", True)]:
            if val is None:
                tk.Label(arch_hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(
                    arch_hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                    bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                    activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                    command=lambda v=val: [x.set(v) for x in self._archetype_vars.values()],
                ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        self._arch_grid = tk.Frame(panel, bg=BG_SETTINGS)
        self._arch_grid.pack(fill=tk.X)
        arch_grid = self._arch_grid
        cols = 5
        _archetypes = load_archetypes()
        for idx, a in enumerate(_archetypes):
            var = tk.BooleanVar(value=True)
            self._archetype_vars[a["id"]] = var
            row, col = divmod(idx, cols)
            tk.Checkbutton(
                arch_grid, text=a["name"], variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
        self._arch_grid_count = len(_archetypes)

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        # ── focal characters ───────────────────────────────────
        tk.Label(panel, text="FOCAL CHARACTERS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w").pack(fill=tk.X)
        tk.Label(panel, text="comma-separated · leave blank to detect automatically",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w"
                 ).pack(fill=tk.X, pady=(2, 4))
        tk.Entry(panel, textvariable=self._focal_chars_var, font=FONT_SMALL,
                 bg=BG_RESULTS, fg=TEXT, relief=tk.FLAT, insertbackground=TEXT,
                 highlightthickness=1, highlightbackground=BTN_BORDER,
                 ).pack(fill=tk.X)
        return panel

    def _build_emot_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X)
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_emot,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        emot_hdr = tk.Frame(panel, bg=BG_SETTINGS)
        emot_hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(emot_hdr, text="EMOTIONS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        tk.Button(
            emot_hdr, text="+ add", font=FONT_SMALL, relief=tk.FLAT,
            bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
            activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
            command=self._open_add_emotion_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        for label, val in [("none", False), ("\xb7", None), ("all", True)]:
            if val is None:
                tk.Label(emot_hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(
                    emot_hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                    bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                    activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                    command=lambda v=val: [x.set(v) for x in self._emotion_vars.values()],
                ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        self._emot_grid = tk.Frame(panel, bg=BG_SETTINGS)
        self._emot_grid.pack(fill=tk.X)
        emot_grid = self._emot_grid
        cols = 5
        _emotions = load_emotions()
        for idx, e in enumerate(_emotions):
            var = tk.BooleanVar(value=True)
            self._emotion_vars[e["id"]] = var
            row, col = divmod(idx, cols)
            tk.Checkbutton(
                emot_grid, text=e["name"], variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
        self._emot_grid_count = len(_emotions)

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        tk.Label(panel, text="FOCAL CHARACTERS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w").pack(fill=tk.X)
        tk.Label(panel, text="comma-separated \xb7 leave blank to detect automatically",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w"
                 ).pack(fill=tk.X, pady=(2, 4))
        tk.Entry(panel, textvariable=self._focal_chars_emot_var, font=FONT_SMALL,
                 bg=BG_RESULTS, fg=TEXT, relief=tk.FLAT, insertbackground=TEXT,
                 highlightthickness=1, highlightbackground=BTN_BORDER,
                 ).pack(fill=tk.X)
        return panel

    def _toggle_copy_settings(self):
        if self._copy_settings_open:
            self._copy_settings.pack_forget()
            self._cog_copy.config(fg=TEXT_MUTED, bg=BG)
        else:
            self._copy_settings.pack(fill=tk.X, before=self._results_outer)
            self._cog_copy.config(fg=ACCENT, bg=BTN_SEC)
        self._copy_settings_open = not self._copy_settings_open

    def _toggle_arch_settings(self):
        if self._arch_settings_open:
            self._arch_settings.pack_forget()
            self._cog_arch.config(fg=TEXT_MUTED, bg=BG)
        else:
            self._arch_settings.pack(fill=tk.X, before=self._char_outer)
            self._cog_arch.config(fg=ACCENT, bg=BTN_SEC)
        self._arch_settings_open = not self._arch_settings_open

    def _toggle_emot_settings(self):
        if self._emot_settings_open:
            self._emot_settings.pack_forget()
            self._cog_emot.config(fg=TEXT_MUTED, bg=BG)
        else:
            self._emot_settings.pack(fill=tk.X, before=self._emot_outer)
            self._cog_emot.config(fg=ACCENT, bg=BTN_SEC)
        self._emot_settings_open = not self._emot_settings_open

    def _build_pers_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X)
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_pers,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        pers_hdr = tk.Frame(panel, bg=BG_SETTINGS)
        pers_hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(pers_hdr, text="TRAITS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        tk.Button(
            pers_hdr, text="+ add", font=FONT_SMALL, relief=tk.FLAT,
            bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
            activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
            command=self._open_add_personality_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        for label, val in [("none", False), ("·", None), ("all", True)]:
            if val is None:
                tk.Label(pers_hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(
                    pers_hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                    bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                    activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                    command=lambda v=val: [x.set(v) for x in self._personality_vars.values()],
                ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        self._pers_grid = tk.Frame(panel, bg=BG_SETTINGS)
        self._pers_grid.pack(fill=tk.X)
        cols = 5
        _traits = load_personality_traits()
        for idx, t in enumerate(_traits):
            var = tk.BooleanVar(value=True)
            self._personality_vars[t["id"]] = var
            row, col = divmod(idx, cols)
            tk.Checkbutton(
                self._pers_grid, text=t["name"], variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
        self._pers_grid_count = len(_traits)

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        tk.Label(panel, text="FOCAL CHARACTERS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w").pack(fill=tk.X)
        tk.Label(panel, text="comma-separated · leave blank to detect automatically",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w"
                 ).pack(fill=tk.X, pady=(2, 4))
        tk.Entry(panel, textvariable=self._focal_chars_pers_var, font=FONT_SMALL,
                 bg=BG_RESULTS, fg=TEXT, relief=tk.FLAT, insertbackground=TEXT,
                 highlightthickness=1, highlightbackground=BTN_BORDER,
                 ).pack(fill=tk.X)
        return panel

    def _toggle_pers_settings(self):
        if self._pers_settings_open:
            self._pers_settings.pack_forget()
            self._cog_pers.config(fg=TEXT_MUTED, bg=BG)
        else:
            self._pers_settings.pack(fill=tk.X, before=self._pers_outer)
            self._cog_pers.config(fg=ACCENT, bg=BTN_SEC)
        self._pers_settings_open = not self._pers_settings_open

    def _build_genre_settings(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_SETTINGS, padx=18, pady=12)

        model_row = tk.Frame(panel, bg=BG_SETTINGS)
        model_row.pack(fill=tk.X)
        tk.Label(model_row, text="MODEL", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        ttk.Combobox(
            model_row, textvariable=self._model_var_genre,
            values=list(MODELS.keys()), state="readonly",
            width=14, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tk.Frame(panel, bg=BORDER, height=1).pack(fill=tk.X, pady=(10, 10))

        genre_hdr = tk.Frame(panel, bg=BG_SETTINGS)
        genre_hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(genre_hdr, text="GENRES", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.LEFT)
        tk.Button(
            genre_hdr, text="+ add", font=FONT_SMALL, relief=tk.FLAT,
            bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
            activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
            command=self._open_add_genre_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        for label, val in [("none", False), ("\xb7", None), ("all", True)]:
            if val is None:
                tk.Label(genre_hdr, text=label, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_SETTINGS).pack(side=tk.RIGHT)
            else:
                tk.Button(
                    genre_hdr, text=label, font=FONT_SMALL, relief=tk.FLAT,
                    bg=BG_SETTINGS, fg=ACCENT, cursor="hand2", bd=0,
                    activebackground=BG_SETTINGS, activeforeground=ACCENT_HOV,
                    command=lambda v=val: [x.set(v) for x in self._genre_vars.values()],
                ).pack(side=tk.RIGHT, padx=(0 if label == "none" else 4, 0))

        self._genre_grid = tk.Frame(panel, bg=BG_SETTINGS)
        self._genre_grid.pack(fill=tk.X)
        genre_grid = self._genre_grid
        cols = 4
        _genres = load_genres()
        for idx, g in enumerate(_genres):
            var = tk.BooleanVar(value=g["name"] != "Literary Fiction")
            self._genre_vars[g["name"]] = var
            row, col = divmod(idx, cols)
            tk.Checkbutton(
                genre_grid, text=g["name"], variable=var,
                font=FONT_SMALL, bg=BG_SETTINGS, fg=TEXT,
                activebackground=BG_SETTINGS, activeforeground=TEXT,
                selectcolor=BG_SETTINGS, anchor="w", cursor="hand2",
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=1)
        self._genre_grid_count = len(_genres)

        return panel

    def _toggle_genre_settings(self):
        if self._genre_settings_open:
            self._genre_settings.pack_forget()
            self._cog_genre.config(fg=TEXT_MUTED, bg=BG)
        else:
            self._genre_settings.pack(fill=tk.X, before=self._genre_outer)
            self._cog_genre.config(fg=ACCENT, bg=BTN_SEC)
        self._genre_settings_open = not self._genre_settings_open

    def _on_style_change(self, key: str):
        cfg = load_style_config()
        cfg[key]["default"] = self._style_vars[key].get()
        save_style_config(cfg)

    # ── icon / logo ────────────────────────────────────────────────────────────

    def _load_icon(self):
        if not ICON_PATH.exists():
            return
        try:
            img = Image.open(ICON_PATH).convert("RGBA")
            self._logo_img_large = ImageTk.PhotoImage(
                img.resize((108, 108), Image.LANCZOS)
            )
            taskbar = ImageTk.PhotoImage(img.resize((64, 64), Image.LANCZOS))
            self._taskbar_icon = taskbar
            self.iconphoto(True, taskbar)
        except Exception:
            pass

    # ── canvas helpers ─────────────────────────────────────────────────────────

    def _on_frame_resize(self, _):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, widget, canvas):
        widget.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        for child in widget.winfo_children():
            self._bind_mousewheel(child, canvas)

    def _clear_results(self):
        for w in self._open_frame.winfo_children():
            w.destroy()
        for w in self._syn_frame.winfo_children():
            w.destroy()
        for w in self._results_frame.winfo_children():
            w.destroy()
        for w in self._char_frame.winfo_children():
            w.destroy()
        for w in self._emot_frame.winfo_children():
            w.destroy()
        for w in self._genre_frame.winfo_children():
            w.destroy()

    def _show_placeholder(self):
        tk.Label(self._open_frame,
                 text="Click Analyze to read the opening and annotate first impressions.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        tk.Label(self._syn_frame,
                 text="Click Analyze to build a synopsis of this manuscript.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        tk.Label(self._results_frame,
                 text="Click Analyze to copyedit this manuscript.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        tk.Label(self._char_frame,
                 text="Click Analyze to identify characters and map archetypes.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        tk.Label(self._emot_frame,
                 text="Click Analyze to map character emotions across the manuscript.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        tk.Label(self._genre_frame,
                 text="Click Analyze to detect genre signals across the manuscript.",
                 font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)

    # ── file open ──────────────────────────────────────────────────────────────

    def _delete_caches(self):
        if not self._filepath:
            return
        from core.cache import _cache_path, CACHE_DIR
        kinds = ("copyedit", "archetypes", "emotions", "personality", "genres", "synopsis", "opening", "arcs")
        existing = [_cache_path(self._filepath, k) for k in kinds
                    if _cache_path(self._filepath, k).exists()]
        if not existing:
            popup = tk.Toplevel(self); popup.title("No caches"); popup.configure(bg=BG)
            popup.resizable(False, False); popup.transient(self); popup.grab_set()
            tk.Label(popup, text="No cache files found for this document.",
                     font=FONT_BODY, fg=TEXT, bg=BG).pack(padx=24, pady=(20, 6))
            _flat_btn(popup, "OK", popup.destroy).pack(pady=(0, 16))
            return

        popup = tk.Toplevel(self); popup.title("Delete caches"); popup.configure(bg=BG)
        popup.resizable(False, False); popup.transient(self); popup.grab_set()
        tk.Label(popup, text=f"Delete {len(existing)} cache file(s) for:",
                 font=FONT_BODY, fg=TEXT, bg=BG).pack(padx=24, pady=(20, 4))
        tk.Label(popup, text=Path(self._filepath).stem,
                 font=FONT_TYPE, fg=ACCENT, bg=BG).pack(padx=24, pady=(0, 4))
        tk.Label(popup, text=", ".join(p.stem.rsplit(".", 1)[-1] for p in existing),
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG).pack(padx=24, pady=(0, 16))

        btn_row = tk.Frame(popup, bg=BG); btn_row.pack(padx=24, pady=(0, 16))

        def _confirm():
            for p in existing:
                try: p.unlink()
                except Exception: pass
            self._synopsis_result     = None
            self._arcs_result         = None
            self._opening_result      = None
            self._results             = []
            self._sheets              = []
            self._emotion_sheets      = []
            self._personality_sheets  = []
            self._clear_results()
            self._show_placeholder()
            self._status_opening.config(text="")
            self._status_copy.config(text="")
            self._status_arch.config(text="")
            self._status_emot.config(text="")
            self._status_pers.config(text="")
            self._status_genre.config(text="")
            self._status_synopsis.config(text="")
            self._btn_save_opening.pack_forget()
            self._btn_save.pack_forget()
            self._btn_save_arch.pack_forget()
            self._btn_save_emot.pack_forget()
            self._btn_save_pers.pack_forget()
            self._btn_save_syn.pack_forget()
            popup.destroy()

        _flat_btn(btn_row, "Delete", _confirm, primary=True).pack(side=tk.LEFT, padx=(0, 8))
        _flat_btn(btn_row, "Cancel", popup.destroy).pack(side=tk.LEFT)

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select manuscript",
            filetypes=[("Word documents", "*.docx"), ("All files", "*.*")],
        )
        if not path:
            return
        self._filepath            = path
        self._results             = []
        self._sheets              = []
        self._emotion_sheets      = []
        self._personality_sheets  = []
        self._synopsis_result     = None
        self._arcs_result         = None
        self._opening_result      = None
        self._file_lbl.config(text=Path(path).stem)
        self._btn_analyze_opening.config(state=tk.NORMAL)
        self._btn_analyze_copy.config(state=tk.NORMAL)
        self._btn_analyze_arch.config(state=tk.NORMAL)
        self._btn_analyze_emot.config(state=tk.NORMAL)
        self._btn_analyze_pers.config(state=tk.NORMAL)
        self._btn_analyze_genre.config(state=tk.NORMAL)
        self._btn_regen_synopsis.config(state=tk.NORMAL)
        self._btn_save_opening.pack_forget()
        self._btn_save.pack_forget()
        self._btn_save_arch.pack_forget()
        self._btn_save_emot.pack_forget()
        self._btn_save_pers.pack_forget()
        self._btn_save_syn.pack_forget()
        self._clear_results()
        self._show_placeholder()
        self._status_opening.config(text="")
        self._status_copy.config(text="")
        self._status_arch.config(text="")
        self._status_emot.config(text="")
        self._status_pers.config(text="")
        self._status_genre.config(text="")
        self._status_synopsis.config(text="")
        if self._landing.winfo_ismapped():
            self._landing.pack_forget()
            self._post_open.pack(fill=tk.BOTH, expand=True)
        self.after(0, lambda: self._start_opening(from_cache_only=True))
        self.after(0, lambda: self._start_synopsis(from_cache_only=True))
        self.after(0, lambda: self._start_copyedit(from_cache_only=True))
        self.after(0, lambda: self._start_archetypes(from_cache_only=True))
        self.after(0, lambda: self._start_emotions(from_cache_only=True))
        self.after(0, lambda: self._start_personality(from_cache_only=True))
        self.after(0, lambda: self._start_genre(from_cache_only=True))

    # ── opening analysis ───────────────────────────────────────────────────────

    def _start_opening(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_opening_cache(self._filepath)
            if cached is not None:
                self._opening_result = cached["result"]
                self._show_opening(cached["result"])
                self._status_opening.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        for w in self._open_frame.winfo_children():
            w.destroy()
        self._btn_analyze_opening.config(state=tk.DISABLED)
        self._btn_save_opening.pack_forget()
        self._status_opening.config(text="Reading opening…")
        threading.Thread(target=self._run_opening, daemon=True).start()

    def _run_opening(self):
        try:
            model         = MODELS[self._model_var_opening.get()]
            enabled_items = [it for it, v in self._opening_item_vars.items() if v.get()] or None
            result, _     = analyze_opening(
                self._filepath,
                model=model,
                enabled_items=enabled_items,
                on_progress=lambda _phase, c, t: self.after(
                    0, self._status_opening.config,
                    {"text": f"Reading opening… {round(c / max(t, 1) * 100)}%"},
                ),
            )
            save_opening_cache(self._filepath, result)
            self._opening_result = result
            self.after(0, self._show_opening, result)
        except Exception as exc:
            self.after(0, self._status_opening.config, {"text": f"Error: {exc}"})
            self.after(0, self._btn_analyze_opening.config, {"state": tk.NORMAL})

    def _show_opening(self, result: OpeningResult):
        for w in self._open_frame.winfo_children():
            w.destroy()

        if not result.items:
            tk.Label(self._open_frame, text="No results returned.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
            self._btn_analyze_opening.config(state=tk.NORMAL)
            self._status_opening.config(text="Done.")
            return

        outer = tk.Frame(self._open_frame, bg=BG_RESULTS, padx=20, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── cutoff snippet ─────────────────────────────────────────────────────
        try:
            from core.reviewer import load_docx
            from core.opening_analyzer import OPENING_WORDS
            import re as _re
            full_text = load_docx(self._filepath)
            excerpt = " ".join(full_text.split()[:OPENING_WORDS])
            # split on sentence-ending punctuation, keep delimiters
            sentences = _re.split(r'(?<=[.!?])\s+', excerpt.strip())
            if len(sentences) >= 2:
                snippet = sentences[-2] + " " + sentences[-1]
            else:
                snippet = sentences[-1] if sentences else ""
            # take last ~180 chars to approximate "one and a half sentences"
            if len(snippet) > 180:
                snippet = snippet[-180:].lstrip()
            tk.Label(outer, text=f"…{snippet}", font=FONT_QUOTE,
                     fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w",
                     justify=tk.LEFT, wraplength=700,
                     ).pack(fill=tk.X, pady=(0, 14))
            tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 10))
        except Exception:
            pass

        # pixel-exact columns via columnconfigure(minsize) inside each row frame
        C1, C2 = 170, 460   # px: checklist, guess

        def _make_row(parent, bg):
            row = tk.Frame(parent, bg=bg)
            row.pack(fill=tk.X)
            row.columnconfigure(0, minsize=C1)
            row.columnconfigure(1, minsize=C2)
            row.rowconfigure(0, weight=1)
            return row

        # header
        hdr = _make_row(outer, BG_SETTINGS)
        tk.Label(hdr, text="Checklist", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w",
                 ).grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        tk.Label(hdr, text="Reader Guess", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w",
                 ).grid(row=0, column=1, sticky="nsew", padx=(12, 0), pady=4)
        tk.Label(hdr, text="Confidence", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_SETTINGS, anchor="w",
                 ).grid(row=0, column=2, sticky="nsew", padx=(12, 4), pady=4)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 2))

        row_colors = [BG_RESULTS, "#f5f0e8"]
        for idx, item in enumerate(result.items):
            bg = row_colors[idx % 2]
            row = _make_row(outer, bg)

            tk.Label(row, text=item.item, font=FONT_TYPE,
                     fg=TEXT_TYPE, bg=bg, anchor="nw",
                     justify=tk.LEFT, wraplength=C1 - 8,
                     ).grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=6)

            tk.Label(row, text=item.guess, font=FONT_BODY,
                     fg=TEXT, bg=bg, anchor="nw",
                     justify=tk.LEFT, wraplength=C2 - 8,
                     ).grid(row=0, column=1, sticky="nsew", padx=(12, 0), pady=6)

            emoji = CONFIDENCE_EMOJI.get(item.confidence, "\U0001f610")
            tk.Label(row, text=emoji, font=("Segoe UI Emoji", 16),
                     fg=TEXT_MUTED, bg=bg, anchor="nw",
                     ).grid(row=0, column=2, sticky="nsew", padx=(12, 4), pady=6)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill=tk.X, pady=(2, 0))

        self._bind_mousewheel(self._open_frame, self._open_canvas)
        self._btn_analyze_opening.config(state=tk.NORMAL)
        self._btn_save_opening.pack(side=tk.LEFT, padx=(12, 0),
                                    in_=self._btn_analyze_opening.master)
        self._status_opening.config(text="Done.")

    def _save_opening_report(self):
        if not self._opening_result or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_opening.pdf"
        self._btn_save_opening.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        def do_save():
            try:
                _write_opening_report(out_path, Path(self._filepath).stem,
                                      self._opening_result)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_opening.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    def _save_syn_report(self):
        if not self._synopsis_result or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_plot.pdf"
        self._btn_save_syn.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        def do_save():
            try:
                _write_plot_report(out_path, Path(self._filepath).stem,
                                   self._synopsis_result, self._arcs_result)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_syn.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    # ── synopsis analysis ──────────────────────────────────────────────────────

    def _start_synopsis(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_synopsis_cache(self._filepath)
            if cached is not None:
                self._synopsis_result = cached["result"]
                arc_cached = load_arcs_cache(self._filepath)
                if arc_cached is not None:
                    self._arcs_result = arc_cached["result"]
                self._show_synopsis(cached["result"])
                self._status_synopsis.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        for w in self._syn_frame.winfo_children():
            w.destroy()
        self._btn_regen_synopsis.config(state=tk.DISABLED)
        self._status_synopsis.config(text="Analyzing plot…")
        threading.Thread(target=self._run_synopsis, daemon=True).start()

    def _run_synopsis(self):
        try:
            model = MODELS[self._model_var_synopsis.get()]
            result, _ = analyze_synopsis(
                self._filepath,
                model=model,
                on_progress=lambda _phase, c, t: self.after(
                    0, self._status_synopsis.config,
                    {"text": f"Analyzing plot… {round(c / max(t, 1) * 100)}%"},
                ),
            )
            save_synopsis_cache(self._filepath, result)
            self._synopsis_result = result

            self.after(0, self._status_synopsis.config, {"text": "Analyzing arcs…"})
            char_names = []
            if self._synopsis_result:
                char_names = [
                    s.name for s in getattr(self, "_sheets", [])
                ] or []
            enabled_structural = [
                arc for arc in load_structural_arcs()
                if self._arc_structural_vars.get(arc["id"], tk.BooleanVar(value=True)).get()
            ] or None
            include_char = self._arc_character_var.get()
            arcs_result, _ = analyze_arcs(
                result,
                model=model,
                enabled_structural=enabled_structural,
                include_character=include_char,
                character_names=char_names,
                on_progress=lambda _phase, c, t: self.after(
                    0, self._status_synopsis.config,
                    {"text": f"Analyzing arcs… {round(c / max(t, 1) * 100)}%"},
                ),
            )
            save_arcs_cache(self._filepath, arcs_result)
            self._arcs_result = arcs_result
            self.after(0, self._show_synopsis, result)
        except Exception as exc:
            self.after(0, self._status_synopsis.config, {"text": f"Error: {exc}"})
            self.after(0, self._btn_regen_synopsis.config, {"state": tk.NORMAL})

    def _show_synopsis(self, result: SynopsisResult):
        for w in self._syn_frame.winfo_children():
            w.destroy()

        # ── full-width header: logline + synopsis ──────────────────────────────
        tk.Frame(self._syn_frame, bg=BG_RESULTS, height=18).pack()
        tk.Label(
            self._syn_frame, text=result.logline,
            font=("Georgia", 13, "italic"), fg=ACCENT, bg=BG_RESULTS,
            wraplength=740, justify=tk.LEFT, anchor="w",
        ).pack(fill=tk.X, padx=20)

        tk.Frame(self._syn_frame, bg=BORDER, height=1).pack(fill=tk.X, padx=20, pady=(14, 0))

        tk.Frame(self._syn_frame, bg=BG_RESULTS, height=10).pack()
        tk.Label(
            self._syn_frame, text=result.synopsis,
            font=FONT_BODY, fg=TEXT, bg=BG_RESULTS,
            wraplength=740, justify=tk.LEFT, anchor="nw",
        ).pack(fill=tk.X, padx=20, pady=(4, 0))

        tk.Frame(self._syn_frame, bg=BORDER, height=1).pack(fill=tk.X, padx=20, pady=(18, 0))

        # ── two-column body: Beats (left) | Arcs (right) ───────────────────────
        columns = tk.Frame(self._syn_frame, bg=BG_RESULTS)
        columns.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 0))
        columns.columnconfigure(0, weight=3)
        columns.columnconfigure(1, weight=2)

        # ── LEFT: Beats ────────────────────────────────────────────────────────
        beats_col = tk.Frame(columns, bg=BG_RESULTS)
        beats_col.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

        tk.Label(beats_col, text="BEATS", font=FONT_LABEL,
                 fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(fill=tk.X, pady=(0, 10))

        if result.plot_beats:
            for idx, beat in enumerate(result.plot_beats):
                last = idx == len(result.plot_beats) - 1
                row = tk.Frame(beats_col, bg=BG_RESULTS)
                row.pack(fill=tk.X)

                left = tk.Frame(row, bg=BG_RESULTS, width=22)
                left.pack(side=tk.LEFT, fill=tk.Y)
                left.pack_propagate(False)

                dot_c = tk.Canvas(left, width=22, height=18,
                                  bg=BG_RESULTS, highlightthickness=0)
                dot_c.pack()
                dot_c.create_oval(7, 4, 15, 12, fill=ACCENT, outline="")

                if not last:
                    tk.Frame(left, bg=BORDER, width=1).pack(fill=tk.Y, expand=True)

                right = tk.Frame(row, bg=BG_RESULTS)
                right.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           padx=(10, 0), pady=(0, 16 if not last else 4))

                hdr = tk.Frame(right, bg=BG_RESULTS)
                hdr.pack(fill=tk.X)
                tk.Label(hdr, text=beat.term, font=FONT_LABEL,
                         fg=TEXT, bg=BG_RESULTS, anchor="w").pack(side=tk.LEFT)

                detail_frame = tk.Frame(right, bg=BG_RESULTS)
                expanded = [False]
                toggle_lbl = tk.Label(hdr, text="more", font=FONT_SMALL,
                                      fg=ACCENT, bg=BG_RESULTS, cursor="hand2")
                toggle_lbl.pack(side=tk.LEFT, padx=(10, 0))

                def _make_toggle(df=detail_frame, st=expanded, lbl=toggle_lbl):
                    def _toggle(_event=None):
                        if st[0]:
                            df.pack_forget()
                            lbl.config(text="more")
                            st[0] = False
                        else:
                            df.pack(fill=tk.X, pady=(6, 0))
                            lbl.config(text="less")
                            st[0] = True
                        self._syn_canvas.configure(
                            scrollregion=self._syn_canvas.bbox("all"))
                    return _toggle

                toggle_lbl.bind("<Button-1>", _make_toggle())

                tk.Label(right, text=beat.summary, font=FONT_SMALL,
                         fg=TEXT_MUTED, bg=BG_RESULTS,
                         wraplength=360, justify=tk.LEFT, anchor="w",
                         ).pack(fill=tk.X, pady=(2, 0))
                if beat.detail:
                    tk.Label(detail_frame, text=beat.detail, font=FONT_SMALL,
                             fg=TEXT, bg=BG_RESULTS,
                             wraplength=360, justify=tk.LEFT, anchor="nw",
                             ).pack(fill=tk.X)
        else:
            tk.Label(beats_col, text="No beats detected.",
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS).pack(anchor="w")

        # ── RIGHT: Arcs ────────────────────────────────────────────────────────
        arcs_col = tk.Frame(columns, bg=BG_RESULTS)
        arcs_col.grid(row=0, column=1, sticky="nsew")

        arcs = self._arcs_result

        if arcs and (arcs.structural or arcs.character):
            if arcs.structural:
                tk.Label(arcs_col, text="ARCS", font=FONT_LABEL,
                         fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(
                    fill=tk.X, pady=(0, 8))
                for arc in arcs.structural:
                    self._arc_card(arcs_col, arc)

            if arcs.structural and arcs.character:
                tk.Frame(arcs_col, bg=BORDER, height=1).pack(fill=tk.X, pady=(8, 0))

            if arcs.character:
                tk.Label(arcs_col, text="CHARACTER", font=FONT_LABEL,
                         fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(
                    fill=tk.X, pady=(10, 8))
                for arc in arcs.character:
                    self._arc_card(arcs_col, arc)
        else:
            tk.Label(arcs_col, text="ARCS", font=FONT_LABEL,
                     fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(fill=tk.X, pady=(0, 10))
            tk.Frame(arcs_col, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 10))
            tk.Label(arcs_col, text="Run Analyze to detect arcs.",
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(anchor="w")

        tk.Frame(self._syn_frame, bg=BG_RESULTS, height=20).pack()
        self._bind_mousewheel(self._syn_frame, self._syn_canvas)
        self._btn_regen_synopsis.config(state=tk.NORMAL)
        self._btn_save_syn.pack(side=tk.LEFT, padx=(8, 0), after=self._status_synopsis)
        self._status_synopsis.config(text="Done.")

    def _arc_card(self, parent, arc: "ArcResult"):
        row = tk.Frame(parent, bg=BG_RESULTS)
        row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(row, text=arc.name, font=FONT_LABEL,
                 fg=TEXT, bg=BG_RESULTS, anchor="w").pack(anchor="w")
        tk.Label(row, text=f"  • {arc.start}  →  {arc.end}",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS,
                 anchor="w", justify=tk.LEFT, wraplength=280).pack(anchor="w", pady=(2, 0))

    # ── copyedit analysis ──────────────────────────────────────────────────────

    def _start_copyedit(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_copyedit_cache(self._filepath)
            if cached is not None:
                self._show(cached["results"])
                self._status_copy.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        self._btn_analyze_copy.config(state=tk.DISABLED)
        self._btn_save.pack_forget()
        self._results = []
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._status_copy.config(text="Starting…")
        threading.Thread(target=self._run_copyedit, daemon=True).start()

    def _run_copyedit(self):
        try:
            model   = MODELS[self._model_var_copy.get()]
            enabled = {t for t, v in self._enabled_vars.items() if v.get()} or None
            results, _ = review_document(
                self._filepath,
                on_progress=lambda c, t: self.after(
                    0, self._status_copy.config, {"text": f"Reviewing manuscript… {round(c/t*100)}%"}
                ),
                model=model,
                enabled_types=enabled,
            )
            save_copyedit_cache(self._filepath, results)
            self.after(0, self._show, results)
        except Exception as exc:
            self.after(0, self._show_error_copy, exc)

    # ── archetype analysis ─────────────────────────────────────────────────────

    def _start_archetypes(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_arch_cache(self._filepath)
            if cached is not None:
                custom = cached.get("custom_archetypes", []) or self._infer_custom_from_cache("archetype", cached)
                self._restore_custom_items("archetype", custom)
                self._show_characters(cached["sheets"])
                self._status_arch.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        self._btn_analyze_arch.config(state=tk.DISABLED)
        self._btn_save_arch.pack_forget()
        self._sheets = []
        for w in self._char_frame.winfo_children():
            w.destroy()
        self._status_arch.config(text="Starting…")
        threading.Thread(target=self._run_archetypes, daemon=True).start()

    def _ensure_synopsis(self, status_fn, model: str) -> None:
        """Run synopsis+arcs pipeline if no cache exists. Updates status_fn with progress."""
        if load_synopsis_cache(self._filepath) is not None:
            return
        PLOT_WEIGHT = 0.25  # synopsis counts as 25% of the total progress bar

        def _plot_prog(_phase, c, t):
            pct = round(c / max(t, 1) * 100 * PLOT_WEIGHT)
            self.after(0, status_fn, {"text": f"Prepending Plot Analysis… {pct}%"})

        syn_model = MODELS[self._model_var_synopsis.get()]
        result, _ = analyze_synopsis(self._filepath, model=syn_model, on_progress=_plot_prog)
        save_synopsis_cache(self._filepath, result)
        self._synopsis_result = result

        try:
            char_names = [s.name for s in getattr(self, "_sheets", [])]
            enabled_structural = [
                arc for arc in load_structural_arcs()
                if self._arc_structural_vars.get(arc["id"], tk.BooleanVar(value=True)).get()
            ] or None
            arcs_result, _ = analyze_arcs(
                result, model=syn_model,
                enabled_structural=enabled_structural,
                include_character=self._arc_character_var.get(),
                character_names=char_names,
            )
            save_arcs_cache(self._filepath, arcs_result)
            self._arcs_result = arcs_result
        except Exception:
            pass

        self.after(0, self._show_synopsis, result)

    def _run_archetypes(self):
        try:
            model = MODELS[self._model_var_arch.get()]
            self._ensure_synopsis(self._status_arch.config, model)
            raw_focal = self._focal_chars_var.get().strip()
            focal = [n.strip() for n in raw_focal.split(",") if n.strip()] if raw_focal else None
            if not focal:
                emot_cached = load_emotion_cache(self._filepath)
                if emot_cached and emot_cached.get("sheets"):
                    focal = [s.name for s in emot_cached["sheets"]]
                    self.after(0, self._status_arch.config, {"text": "Reusing characters from emotion cache…"})
            enabled_arch = [aid for aid, v in self._archetype_vars.items() if v.get()] or None
            syn_text = self._synopsis_result.synopsis if self._synopsis_result else ""
            sheets, _ = analyze_characters(
                self._filepath,
                syn_text,
                model=model,
                focal_characters=focal,
                enabled_archetypes=enabled_arch,
                extra_archetypes=self._custom_archetypes or None,
                on_progress=lambda c, t: self.after(
                    0, self._status_arch.config,
                    {"text": f"Analyzing manuscript… {round(c/t*100)}%"},
                ),
            )
            save_arch_cache(self._filepath, sheets, self._custom_archetypes or None)
            self.after(0, self._show_characters, sheets)
        except Exception as exc:
            self.after(0, self._show_error_arch, exc)

    # ── emotion analysis ───────────────────────────────────────────────────────

    def _start_emotions(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_emotion_cache(self._filepath)
            if cached is not None:
                custom = cached.get("custom_emotions", []) or self._infer_custom_from_cache("emotion", cached)
                self._restore_custom_items("emotion", custom)
                self._show_emotion_results(cached["sheets"])
                self._status_emot.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        self._btn_analyze_emot.config(state=tk.DISABLED)
        self._btn_save_emot.pack_forget()
        self._emotion_sheets = []
        for w in self._emot_frame.winfo_children():
            w.destroy()
        self._status_emot.config(text="Starting…")
        threading.Thread(target=self._run_emotions, daemon=True).start()

    def _run_emotions(self):
        try:
            model = MODELS[self._model_var_emot.get()]
            self._ensure_synopsis(self._status_emot.config, model)
            raw_focal = self._focal_chars_emot_var.get().strip()
            focal = [n.strip() for n in raw_focal.split(",") if n.strip()] if raw_focal else None
            if not focal:
                arch_cached = load_arch_cache(self._filepath)
                if arch_cached and arch_cached.get("sheets"):
                    focal = [s.name for s in arch_cached["sheets"]]
                    self.after(0, self._status_emot.config, {"text": "Reusing characters from archetype cache…"})
            enabled_emot = [eid for eid, v in self._emotion_vars.items() if v.get()] or None
            syn_text = self._synopsis_result.synopsis if self._synopsis_result else ""
            sheets, _ = analyze_emotions(
                self._filepath,
                syn_text,
                model=model,
                focal_characters=focal,
                enabled_emotions=enabled_emot,
                extra_emotions=self._custom_emotions or None,
                on_progress=lambda c, t: self.after(
                    0, self._status_emot.config,
                    {"text": f"Analyzing emotions… {round(c/t*100)}%"},
                ),
            )
            save_emotion_cache(self._filepath, sheets, self._custom_emotions or None)
            self.after(0, self._show_emotion_results, sheets)
        except Exception as exc:
            self.after(0, self._show_error_emot, exc)

    # ── display: copyedit ──────────────────────────────────────────────────────

    def _show(self, results: list[ChunkResult]):
        self._results = results
        for w in self._results_frame.winfo_children():
            w.destroy()

        if not results:
            tk.Label(self._results_frame, text="No issues found.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        else:
            by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for r in results:
                for f in r.flags:
                    by_type[f.type].append((f.text, f.issue))

            total = sum(len(v) for v in by_type.values())
            tk.Label(
                self._results_frame,
                text=f"{total} issue{'s' if total != 1 else ''} · "
                     f"{len(by_type)} categor{'ies' if len(by_type) != 1 else 'y'}",
                font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w"
            ).pack(fill=tk.X, padx=16, pady=(14, 10))

            for etype in sorted(by_type):
                self._add_type_block(etype, by_type[etype])

            self._btn_save.pack(side=tk.LEFT, padx=(8, 0), after=self._status_copy)

        self._bind_mousewheel(self._results_frame, self._canvas)
        self._btn_analyze_copy.config(state=tk.NORMAL)
        self._status_copy.config(text="Done.")

    def _add_type_block(self, etype: str, items: list[tuple[str, str]]):
        block = tk.Frame(self._results_frame, bg=BG_RESULTS)
        block.pack(fill=tk.X, padx=16, pady=(0, 14))

        heading = tk.Frame(block, bg=BG_RESULTS)
        heading.pack(fill=tk.X, pady=(0, 6))
        tk.Label(heading, text=etype,
                 font=FONT_TYPE, fg=TEXT_TYPE, bg=BG_RESULTS, anchor="w").pack(side=tk.LEFT)
        tk.Label(heading, text=f"  {len(items)}",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w").pack(side=tk.LEFT)

        for text, issue in items[:3]:
            item = tk.Frame(block, bg=BG, relief=tk.FLAT, bd=0)
            item.pack(fill=tk.X, pady=2, ipady=8, ipadx=10)
            tk.Frame(item, bg=ACCENT, width=3).pack(side=tk.LEFT, fill=tk.Y)
            inner = tk.Frame(item, bg=BG)
            inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
            tk.Label(inner, text=f"\u201c{text}\u201d",
                     font=FONT_QUOTE, fg=TEXT, bg=BG,
                     anchor="w", wraplength=580, justify=tk.LEFT).pack(fill=tk.X)
            tk.Label(inner, text=issue,
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG,
                     anchor="w", wraplength=580, justify=tk.LEFT).pack(fill=tk.X, pady=(2, 0))

        hidden = len(items) - 3
        if hidden > 0:
            tk.Label(block, text=f"+ {hidden} more",
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w"
                     ).pack(fill=tk.X, pady=(2, 0))

    # ── display: archetypes ────────────────────────────────────────────────────

    def _show_characters(self, sheets: list[CharacterSheet]):
        self._sheets = sheets
        for w in self._char_frame.winfo_children():
            w.destroy()

        if not sheets:
            tk.Label(self._char_frame, text="No named characters detected.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        else:
            tk.Label(
                self._char_frame,
                text=f"{len(sheets)} character{'s' if len(sheets) != 1 else ''} identified",
                font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w"
            ).pack(fill=tk.X, padx=16, pady=(14, 10))

            all_cidxs = [s["chunk_idx"] for sh in sheets for s in sh.signals_ordered]
            tc = max(all_cidxs) + 1 if all_cidxs else 1
            for idx, sheet in enumerate(sheets):
                self._add_character_card(sheet, first=(idx == 0))
                plot_frame = _make_arch_plot_widget(self._char_frame, [sheet], total_chunks=tc)
                if plot_frame is not None:
                    plot_frame.pack(fill=tk.X, padx=16, pady=(0, 14))
            self._btn_save_arch.pack(side=tk.LEFT, padx=(8, 0), after=self._status_arch)

        self._bind_mousewheel(self._char_frame, self._char_canvas)
        self._btn_analyze_arch.config(state=tk.NORMAL)
        self._status_arch.config(text="Done.")

    def _add_character_card(self, sheet: CharacterSheet, first: bool = False):
        card = tk.Frame(self._char_frame, bg=BG, relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=16, pady=(0 if first else 18, 14))
        tk.Frame(card, bg=ACCENT, width=3).pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(card, bg=BG)
        body.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 16), pady=(10, 12))

        # name + aliases
        name_row = tk.Frame(body, bg=BG)
        name_row.pack(fill=tk.X)
        tk.Label(name_row, text=sheet.name,
                 font=FONT_TITLE, fg=TEXT, bg=BG, anchor="w").pack(side=tk.LEFT)
        if sheet.aliases:
            tk.Label(name_row,
                     text="  " + ", ".join(f'"{a}"' for a in sheet.aliases),
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w").pack(side=tk.LEFT)

        # synopsis
        if sheet.synopsis:
            tk.Label(body, text=sheet.synopsis,
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG,
                     anchor="w", wraplength=680, justify=tk.LEFT).pack(fill=tk.X, pady=(3, 0))

        # bar chart + archetype badge icons side by side
        top3 = sheet.top_archetypes[:3]
        if top3:
            chart_h = len(top3) * 26 + 8
            bar_row = tk.Frame(body, bg=BG)
            bar_row.pack(fill=tk.X, pady=(10, 0))

            chart = tk.Canvas(bar_row, bg=BG, height=chart_h, width=520,
                              highlightthickness=0, bd=0)
            chart.pack(side=tk.LEFT)
            chart.bind(
                "<Configure>",
                lambda e, c=chart, d=top3: _draw_arch_bars(c, d, e.width),
            )

            # badge strip — icons sized by softmax of scores, bottom-aligned
            badge_frame = tk.Frame(bar_row, bg=BG)
            badge_frame.pack(side=tk.LEFT, padx=(12, 0), anchor="s")
            badge_refs = []
            bg_rgb = tuple(int(BG[i:i+2], 16) for i in (1, 3, 5))
            arch_scores = [sc for _, _, sc in top3]
            icon_sizes  = _softmax_sizes(arch_scores, lo=52, hi=96)
            for (arch_id, _, _), sz in zip(top3, icon_sizes):
                try:
                    import numpy as _np
                    _raw = Image.open(ARCH_ICON_PATH(arch_id)).convert("RGBA")
                    _bg  = Image.new("RGBA", _raw.size, bg_rgb + (255,))
                    img  = Image.alpha_composite(_bg, _raw).convert("RGB")
                    arr  = _np.array(img)
                    white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
                    arr[white] = bg_rgb
                    img = Image.fromarray(arr.astype(_np.uint8), "RGB")
                    ow, oh = img.size
                    bw = max(1, int(ow * sz / oh))
                    img = img.resize((bw, sz), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    badge_refs.append(photo)
                    tk.Label(badge_frame, image=photo, bg=BG, bd=0).pack(
                        side=tk.LEFT, padx=2, anchor="s")
                except Exception:
                    pass
            badge_frame._badge_refs = badge_refs   # prevent GC

        # signal count
        tk.Label(body, text=f"{sheet.total_signals} signals",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w"
                 ).pack(fill=tk.X, pady=(6, 0))

    # ── display: emotions ──────────────────────────────────────────────────────

    def _show_emotion_results(self, sheets: list[CharacterSheet]):
        self._emotion_sheets = sheets
        for w in self._emot_frame.winfo_children():
            w.destroy()

        if not sheets:
            tk.Label(self._emot_frame, text="No named characters detected.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        else:
            mode = self._emot_group_by.get()

            toggle_row = tk.Frame(self._emot_frame, bg=BG_RESULTS)
            toggle_row.pack(fill=tk.X, padx=16, pady=(14, 2))
            _make_group_toggle(
                toggle_row,
                self._emot_group_by,
                "By Emotion", "emotion",
                "By Character", "character",
                on_change=lambda: self._show_emotion_results(self._emotion_sheets),
            ).pack(side=tk.LEFT)

            tk.Label(
                self._emot_frame,
                text=f"{len(sheets)} character{'s' if len(sheets) != 1 else ''} identified",
                font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w"
            ).pack(fill=tk.X, padx=16, pady=(6, 8))

            all_cidxs = [s["chunk_idx"] for sh in sheets for s in sh.signals_ordered]
            tc = max(all_cidxs) + 1 if all_cidxs else 1

            if mode == "emotion":
                radar = _make_combined_emotion_radar_widget(
                    self._emot_frame, sheets,
                    extra_emotions=self._custom_emotions or None,
                )
                if radar is not None:
                    radar.pack(padx=16, pady=(0, 14))
                plot_frame = _make_emotion_by_group_plot_widget(
                    self._emot_frame, sheets, total_chunks=tc,
                )
                if plot_frame is not None:
                    plot_frame.pack(fill=tk.X, padx=16, pady=(0, 14))
            else:
                for idx, sheet in enumerate(sheets):
                    self._add_emotion_card(sheet, first=(idx == 0))
                    plot_frame = _make_emotion_plot_widget(self._emot_frame, [sheet], total_chunks=tc)
                    if plot_frame is not None:
                        plot_frame.pack(fill=tk.X, padx=16, pady=(0, 14))

            self._btn_save_emot.pack(side=tk.LEFT, padx=(8, 0), after=self._status_emot)

        self._bind_mousewheel(self._emot_frame, self._emot_canvas)
        self._btn_analyze_emot.config(state=tk.NORMAL)
        self._status_emot.config(text="Done.")

    def _add_emotion_card(self, sheet: CharacterSheet, first: bool = False):
        card = tk.Frame(self._emot_frame, bg=BG, relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=16, pady=(0 if first else 18, 14))
        tk.Frame(card, bg=ACCENT, width=3).pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(card, bg=BG)
        body.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 16), pady=(10, 12))

        name_row = tk.Frame(body, bg=BG)
        name_row.pack(fill=tk.X)
        tk.Label(name_row, text=sheet.name,
                 font=FONT_TYPE, fg=TEXT, bg=BG, anchor="w").pack(side=tk.LEFT)
        if sheet.aliases:
            tk.Label(name_row,
                     text="  " + ", ".join(f'"{a}"' for a in sheet.aliases),
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w").pack(side=tk.LEFT)

        if sheet.synopsis:
            tk.Label(body, text=sheet.synopsis,
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG,
                     anchor="w", wraplength=680, justify=tk.LEFT).pack(fill=tk.X, pady=(3, 0))

        # radar + top-3 emotion icons side by side
        top3 = [(eid, ename, sc) for eid, ename, sc in sheet.top_archetypes[:3] if sc > 0]
        radar_row = tk.Frame(body, bg=BG)
        radar_row.pack(anchor="w", pady=(10, 0))

        radar = _make_emotion_radar_widget(radar_row, sheet,
                                           extra_emotions=self._custom_emotions or None)
        if radar is not None:
            radar.pack(side=tk.LEFT)

        if top3:
            icons_strip = tk.Frame(radar_row, bg=BG)
            icons_strip.pack(side=tk.LEFT, padx=(20, 0), anchor="s")
            icon_refs = []
            emot_scores = [sc for _, _, sc in top3]
            icon_sizes  = _softmax_sizes(emot_scores, lo=52, hi=96)
            for (eid, ename, _), sz in zip(top3, icon_sizes):
                icon_path = _emotion_icon_path(eid)
                cell = tk.Frame(icons_strip, bg=BG)
                cell.pack(side=tk.LEFT, padx=(0, 12), anchor="s")
                try:
                    img = Image.open(icon_path).convert("RGBA")
                    img = img.resize((sz, sz), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    icon_refs.append(photo)
                    tk.Label(cell, image=photo, bg=BG, bd=0).pack()
                except Exception:
                    pass
                tk.Label(cell, text=ename, font=FONT_SMALL, fg=TEXT_MUTED,
                         bg=BG, anchor="center").pack()
            icons_strip._icon_refs = icon_refs  # prevent GC

        tk.Label(body, text=f"{sheet.total_signals} emotional moments",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w"
                 ).pack(fill=tk.X, pady=(6, 0))

    # ── save ───────────────────────────────────────────────────────────────────

    def _make_save_popup(self, out_path: str):
        """Create a saving popup. Returns an update(success, extra_line, error) callback."""
        popup = tk.Toplevel(self)
        popup.title("Saving…")
        popup.configure(bg=BG)
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        tk.Label(popup, text=Path(out_path).name,
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG).pack(padx=24, pady=(20, 6))
        status_lbl = tk.Label(popup, text="Saving…", font=FONT_UI, fg=TEXT, bg=BG)
        status_lbl.pack(padx=24, pady=(0, 20))

        btn_row = tk.Frame(popup, bg=BG)

        def _center():
            popup.update_idletasks()
            x = self.winfo_x() + (self.winfo_width()  - popup.winfo_width())  // 2
            y = self.winfo_y() + (self.winfo_height() - popup.winfo_height()) // 2
            popup.geometry(f"+{x}+{y}")

        _center()

        def update(success: bool, extra_line: str = "", error: str = ""):
            if success:
                popup.title("Saved")
                status_lbl.config(text="File saved successfully")
                if extra_line:
                    tk.Label(popup, text=extra_line,
                             font=FONT_SMALL, fg=TEXT_MUTED, bg=BG).pack(padx=24, pady=(0, 4))
                btn_row.pack(padx=24, pady=(12, 20))

                def show_in_explorer():
                    import subprocess
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(out_path)])
                    popup.destroy()

                _flat_btn(btn_row, "Show in Explorer", show_in_explorer).pack(side=tk.LEFT, padx=(0, 8))
                _flat_btn(btn_row, "OK", popup.destroy, primary=True).pack(side=tk.LEFT)
            else:
                popup.title("Error")
                status_lbl.config(text=f"Save failed: {error}", fg="#e05252")
                btn_row.pack(padx=24, pady=(12, 20))
                _flat_btn(btn_row, "OK", popup.destroy, primary=True).pack()
            _center()

        return update

    def _save(self):
        if not self._results or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_reviewed.docx"
        self._btn_save.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        def do_save():
            try:
                anchored = annotate(self._filepath, out_path, self._results)
                total    = sum(len(r.flags) for r in self._results)
                self.after(0, update, True, f"{anchored}/{total} comments anchored")
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    def _save_arch_report(self):
        if not self._sheets or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_archetypes.pdf"
        self._btn_save_arch.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        def do_save():
            try:
                _write_arch_report(out_path, Path(self._filepath).stem, self._sheets)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_arch.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    def _save_emotion_report(self):
        if not self._emotion_sheets or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_emotions.pdf"
        self._btn_save_emot.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        group_by_emot = self._emot_group_by.get()

        def do_save():
            try:
                _write_emotion_report(out_path, Path(self._filepath).stem,
                                      self._emotion_sheets, group_by=group_by_emot)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_emot.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    def _save_personality_report(self):
        if not self._personality_sheets or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_personality.pdf"
        self._btn_save_pers.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        group_by_pers = self._pers_group_by.get()

        def do_save():
            try:
                _write_personality_report(out_path, Path(self._filepath).stem,
                                          self._personality_sheets, group_by=group_by_pers)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_pers.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()

    # ── errors ─────────────────────────────────────────────────────────────────

    def _show_error_copy(self, exc: Exception):
        for w in self._results_frame.winfo_children():
            w.destroy()
        tk.Label(self._results_frame, text=f"Error: {exc}",
                 font=FONT_BODY, fg="#e05252", bg=BG_RESULTS, anchor="w"
                 ).pack(padx=16, pady=20)
        self._status_copy.config(text="Error.")
        self._btn_analyze_copy.config(state=tk.NORMAL)

    def _show_error_arch(self, exc: Exception):
        for w in self._char_frame.winfo_children():
            w.destroy()
        tk.Label(self._char_frame, text=f"Error: {exc}",
                 font=FONT_BODY, fg="#e05252", bg=BG_RESULTS, anchor="w"
                 ).pack(padx=16, pady=20)
        self._status_arch.config(text="Error.")
        self._btn_analyze_arch.config(state=tk.NORMAL)

    def _show_error_emot(self, exc: Exception):
        for w in self._emot_frame.winfo_children():
            w.destroy()
        tk.Label(self._emot_frame, text=f"Error: {exc}",
                 font=FONT_BODY, fg="#e05252", bg=BG_RESULTS, anchor="w"
                 ).pack(padx=16, pady=20)
        self._status_emot.config(text="Error.")
        self._btn_analyze_emot.config(state=tk.NORMAL)

    # ── personality analysis ───────────────────────────────────────────────────

    def _start_personality(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_personality_cache(self._filepath)
            if cached is not None:
                custom = cached.get("custom_traits", []) or self._infer_custom_from_cache("trait", cached)
                self._restore_custom_items("trait", custom)
                self._show_personality_results(cached["sheets"])
                self._status_pers.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        self._btn_analyze_pers.config(state=tk.DISABLED)
        self._btn_save_pers.pack_forget()
        self._personality_sheets = []
        for w in self._pers_frame.winfo_children():
            w.destroy()
        self._status_pers.config(text="Starting…")
        threading.Thread(target=self._run_personality, daemon=True).start()

    def _run_personality(self):
        try:
            model = MODELS[self._model_var_pers.get()]
            self._ensure_synopsis(self._status_pers.config, model)
            raw_focal = self._focal_chars_pers_var.get().strip()
            focal = [n.strip() for n in raw_focal.split(",") if n.strip()] if raw_focal else None
            if not focal:
                arch_cached = load_arch_cache(self._filepath)
                if arch_cached and arch_cached.get("sheets"):
                    focal = [s.name for s in arch_cached["sheets"]]
                    self.after(0, self._status_pers.config, {"text": "Reusing characters from archetype cache…"})
            enabled_traits = [tid for tid, v in self._personality_vars.items() if v.get()] or None
            syn_text = self._synopsis_result.synopsis if self._synopsis_result else ""
            sheets, _ = analyze_personality(
                self._filepath,
                syn_text,
                model=model,
                focal_characters=focal,
                enabled_traits=enabled_traits,
                extra_traits=self._custom_personalities or None,
                on_progress=lambda c, t: self.after(
                    0, self._status_pers.config,
                    {"text": f"Analyzing personality… {round(c/t*100)}%"},
                ),
            )
            save_personality_cache(self._filepath, sheets, self._custom_personalities or None)
            self.after(0, self._show_personality_results, sheets)
        except Exception as exc:
            self.after(0, self._show_error_pers, exc)

    def _show_personality_results(self, sheets: list[CharacterSheet]):
        self._personality_sheets = sheets
        for w in self._pers_frame.winfo_children():
            w.destroy()

        if not sheets:
            tk.Label(self._pers_frame, text="No named characters detected.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        else:
            mode = self._pers_group_by.get()

            toggle_row = tk.Frame(self._pers_frame, bg=BG_RESULTS)
            toggle_row.pack(fill=tk.X, padx=16, pady=(14, 2))
            _make_group_toggle(
                toggle_row,
                self._pers_group_by,
                "By Trait", "trait",
                "By Character", "character",
                on_change=lambda: self._show_personality_results(self._personality_sheets),
            ).pack(side=tk.LEFT)

            tk.Label(
                self._pers_frame,
                text=f"{len(sheets)} character{'s' if len(sheets) != 1 else ''} identified",
                font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w"
            ).pack(fill=tk.X, padx=16, pady=(6, 8))

            all_cidxs = [s["chunk_idx"] for sh in sheets for s in sh.signals_ordered]
            tc = max(all_cidxs) + 1 if all_cidxs else 1

            if mode == "trait":
                radar = _make_combined_personality_radar_widget(
                    self._pers_frame, sheets,
                    extra_traits=self._custom_personalities or None,
                )
                if radar is not None:
                    radar.pack(padx=16, pady=(0, 14))
                plot_frame = _make_personality_by_group_plot_widget(
                    self._pers_frame, sheets, total_chunks=tc,
                )
                if plot_frame is not None:
                    plot_frame.pack(fill=tk.X, padx=16, pady=(0, 14))
            else:
                for idx, sheet in enumerate(sheets):
                    self._add_personality_card(sheet, first=(idx == 0))
                    plot_frame = _make_personality_plot_widget(self._pers_frame, [sheet], total_chunks=tc)
                    if plot_frame is not None:
                        plot_frame.pack(fill=tk.X, padx=16, pady=(0, 14))

            self._btn_save_pers.pack(side=tk.LEFT, padx=(8, 0), after=self._status_pers)

        self._bind_mousewheel(self._pers_frame, self._pers_canvas)
        self._btn_analyze_pers.config(state=tk.NORMAL)
        self._status_pers.config(text="Done.")

    def _add_personality_card(self, sheet: CharacterSheet, first: bool = False):
        card = tk.Frame(self._pers_frame, bg=BG, relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=16, pady=(0 if first else 18, 14))
        tk.Frame(card, bg=ACCENT, width=3).pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(card, bg=BG)
        body.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 16), pady=(10, 12))

        name_row = tk.Frame(body, bg=BG)
        name_row.pack(fill=tk.X)
        tk.Label(name_row, text=sheet.name,
                 font=FONT_TYPE, fg=TEXT, bg=BG, anchor="w").pack(side=tk.LEFT)
        if sheet.aliases:
            tk.Label(name_row,
                     text="  " + ", ".join(f'"{a}"' for a in sheet.aliases),
                     font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w").pack(side=tk.LEFT)

        if sheet.synopsis:
            tk.Label(body, text=sheet.synopsis,
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG,
                     anchor="w", wraplength=680, justify=tk.LEFT).pack(fill=tk.X, pady=(3, 0))

        top3 = [(tid, tname, sc) for tid, tname, sc in sheet.top_archetypes[:3] if abs(sc) >= 0.1]
        radar_row = tk.Frame(body, bg=BG)
        radar_row.pack(anchor="w", pady=(10, 0))

        radar = _make_personality_radar_widget(radar_row, sheet,
                                               extra_traits=self._custom_personalities or None)
        if radar is not None:
            radar.pack(side=tk.LEFT)

        if top3:
            # build pole-name lookup from loaded traits
            _all_traits = load_personality_traits()
            _pole_map = {
                t["id"]: (t.get("high_name", t["name"]), t.get("low_name", t["name"]))
                for t in _all_traits
            }
            icons_strip = tk.Frame(radar_row, bg=BG)
            icons_strip.pack(side=tk.LEFT, padx=(20, 0), anchor="s")
            icon_refs = []
            pers_scores = [abs(sc) for _, _, sc in top3]
            icon_sizes  = _softmax_sizes(pers_scores, lo=52, hi=96)
            for (tid, tname, t_sc), sz in zip(top3, icon_sizes):
                icon_path = _personality_icon_path(tid, t_sc)
                high_n, low_n = _pole_map.get(tid, (tname, tname))
                display_name = high_n if t_sc >= 0 else low_n
                cell = tk.Frame(icons_strip, bg=BG)
                cell.pack(side=tk.LEFT, padx=(0, 12), anchor="s")
                try:
                    img = Image.open(icon_path).convert("RGBA")
                    img = img.resize((sz, sz), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    icon_refs.append(photo)
                    tk.Label(cell, image=photo, bg=BG, bd=0).pack()
                except Exception:
                    pass
                tk.Label(cell, text=display_name, font=FONT_SMALL, fg=TEXT_MUTED,
                         bg=BG, anchor="center").pack()
            icons_strip._icon_refs = icon_refs

        tk.Label(body, text=f"{sheet.total_signals} personality signals",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG, anchor="w"
                 ).pack(fill=tk.X, pady=(6, 0))

    def _show_error_pers(self, exc: Exception):
        for w in self._pers_frame.winfo_children():
            w.destroy()
        tk.Label(self._pers_frame, text=f"Error: {exc}",
                 font=FONT_BODY, fg="#e05252", bg=BG_RESULTS, anchor="w"
                 ).pack(padx=16, pady=20)
        self._status_pers.config(text="Error.")
        self._btn_analyze_pers.config(state=tk.NORMAL)

    # ── genre analysis ─────────────────────────────────────────────────────────

    def _start_genre(self, from_cache_only: bool = False):
        if from_cache_only:
            cached = load_genre_cache(self._filepath)
            if cached is not None:
                custom = cached.get("custom_genres", []) or self._infer_custom_from_cache("genre", cached)
                self._restore_custom_items("genre", custom)
                self._show_genre_results(cached["results"])
                self._status_genre.config(
                    text=f"Cached {_fmt_cache_dt(cached['analyzed_at'])} · Analyze to refresh"
                )
            return
        self._btn_analyze_genre.config(state=tk.DISABLED)
        self._btn_save_genre.pack_forget()
        self._genre_results = []
        for w in self._genre_frame.winfo_children():
            w.destroy()
        self._status_genre.config(text="Starting…")
        threading.Thread(target=self._run_genre, daemon=True).start()

    def _run_genre(self):
        try:
            model = MODELS[self._model_var_genre.get()]
            enabled_genres = (
                [name for name, v in self._genre_vars.items() if v.get()] or None
            )
            results, _ = analyze_genres(
                self._filepath,
                model=model,
                on_progress=lambda phase, c, t: self.after(
                    0, self._status_genre.config,
                    {"text": f"Analyzing genres… {round(c/t*100)}%"},
                ),
                enabled_genres=enabled_genres,
                extra_genres=self._custom_genres or None,
            )
            save_genre_cache(self._filepath, results, self._custom_genres or None)
            self.after(0, self._show_genre_results, results)
        except Exception as exc:
            self.after(0, self._show_error_genre, exc)

    def _show_genre_results(self, results: list[GenreResult]):
        self._genre_results = results
        for w in self._genre_frame.winfo_children():
            w.destroy()

        if not results:
            tk.Label(self._genre_frame, text="No genre signals detected.",
                     font=FONT_BODY, fg=TEXT_MUTED, bg=BG_RESULTS).pack(pady=30)
        else:
            total_signals = sum(r.total_signals for r in results)
            tk.Label(
                self._genre_frame,
                text=f"{total_signals} signals across {len(results)} genre{'s' if len(results) != 1 else ''}",
                font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_RESULTS, anchor="w",
            ).pack(fill=tk.X, padx=16, pady=(14, 8))

            pie = _make_genre_pie_widget(self._genre_frame, results)
            if pie:
                pie.pack(fill=tk.X, padx=16, pady=(0, 8))

            plot = _make_genre_plot_widget(self._genre_frame, results)
            if plot:
                plot.pack(fill=tk.X, padx=16, pady=(0, 16))

            self._btn_save_genre.pack(side=tk.LEFT, padx=(8, 0),
                                      after=self._status_genre)

        self._bind_mousewheel(self._genre_frame, self._genre_canvas)
        self._btn_analyze_genre.config(state=tk.NORMAL)
        self._status_genre.config(text="Done.")

    def _show_error_genre(self, exc: Exception):
        for w in self._genre_frame.winfo_children():
            w.destroy()
        tk.Label(self._genre_frame, text=f"Error: {exc}",
                 font=FONT_BODY, fg="#e05252", bg=BG_RESULTS, anchor="w"
                 ).pack(padx=16, pady=20)
        self._status_genre.config(text="Error.")
        self._btn_analyze_genre.config(state=tk.NORMAL)

    def _save_genre_report(self):
        if not self._genre_results or not self._filepath:
            return
        base, _ = os.path.splitext(self._filepath)
        out_path = base + "_genres.pdf"
        self._btn_save_genre.config(state=tk.DISABLED)
        update = self._make_save_popup(out_path)

        def do_save():
            try:
                _write_genre_report(out_path, Path(self._filepath).stem,
                                    self._genre_results)
                self.after(0, update, True)
            except Exception as exc:
                self.after(0, update, False, "", str(exc))
            finally:
                self.after(0, self._btn_save_genre.config, {"state": tk.NORMAL})

        threading.Thread(target=do_save, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
