"""
Icons for each of the 50 archetypes.

ICON_DIR — directory containing one PNG per archetype ID
           (assets/archeypes/individual/<id>.png)
ICON_PATH — convenience function returning the Path for a given ID
ABBREVS  — 2-letter text fallback for environments without image support

Usage in matplotlib (OffsetImage):
    from config.archetype_icons import ICON_PATH
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    img = plt.imread(str(ICON_PATH(arch_id)))
    box = OffsetImage(img, zoom=0.25)
    ab  = AnnotationBbox(box, (x, y), frameon=False)
    ax.add_artist(ab)

Usage as text fallback:
    from config.archetype_icons import ABBREVS
    label = ABBREVS.get(arch_id, "?")
"""

from pathlib import Path

ICON_DIR = Path(__file__).parent.parent / "assets" / "archeypes" / "individual"


def ICON_PATH(arch_id: str) -> Path:
    """Return the PNG path for an archetype ID (raises if missing)."""
    p = ICON_DIR / f"{arch_id}.png"
    if not p.exists():
        raise FileNotFoundError(f"No icon for archetype '{arch_id}': {p}")
    return p

# ── 2-letter abbreviations (text fallback) ───────────────────────────────────

ABBREVS: dict[str, str] = {
    "hero":              "Hr",
    "mentor":            "Mn",
    "shadow":            "Sh",
    "trickster":         "Tr",
    "golden_retriever":  "GR",
    "brooding_antihero": "BA",
    "innocent":          "In",
    "rebel":             "Rb",
    "sage":              "Sg",
    "caregiver":         "Cg",
    "ruler":             "Rl",
    "explorer":          "Ex",
    "romantic_idealist": "RI",
    "pragmatist":        "Pr",
    "nihilist":          "Ni",
    "shapeshifter":      "SS",
    "jester":            "Js",
    "wounded_warrior":   "WW",
    "fierce_protector":  "FP",
    "survivor":          "Sv",
    "obsessive":         "Ob",
    "seducer":           "Sd",
    "reluctant_chosen":  "RC",
    "gothic_lover":      "GL",
    "true_believer":     "TB",
    "everyman_orphan":   "EO",
    "visionary_creator": "VC",
    "melancholic":       "Ml",
    "bumbler":           "Bu",
    "jock":              "Jk",
    "academic":          "Ac",
    "hermit":            "Hm",
    "social_butterfly":  "SB",
    "hedonist":          "Hd",
    "martyr":            "Mt",
    "perfectionist":     "Pf",
    "coward":            "Cw",
    "cynic":             "Cy",
    "bully":             "Bl",
    "people_pleaser":    "PP",
    "wallflower":        "Wf",
    "entitled":          "En",
    "revolutionary":     "Rv",
    "grifter":           "Gf",
    "contrarian":        "Cn",
    "gossip":            "Go",
    "dreamer":           "Dr",
    "loner":             "Lo",
    "fading_star":       "FS",
    "peacemaker":        "Pm",
}
