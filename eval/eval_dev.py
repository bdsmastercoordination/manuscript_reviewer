"""
Dev-set evaluator. Use this file to tune prompts.
Run: python eval.py
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docx import Document
from core.reviewer import review_document, ChunkResult, TokenUsage

TEST_DOC_PATH = Path(__file__).parent.parent / "test_manuscript.docx"

# ---------------------------------------------------------------------------
# Known errors — chunk assignments assume CHUNK_CHARS = 2000.
# Original sections S1-S14 (43 errors). Extended sections S15-S28 (47 errors).
# ---------------------------------------------------------------------------
KNOWN_ERRORS = [
    # ── chunk 1: spelling ──────────────────────────────────────────────────
    {"id": "spell-1",  "description": "'accomodate' -> 'accommodate'",  "keywords": ["accomodate",  "accommodate"]},
    {"id": "spell-2",  "description": "'relavant' -> 'relevant'",        "keywords": ["relavant",    "relevant"]},
    {"id": "spell-3",  "description": "'occassion' -> 'occasion'",       "keywords": ["occassion",   "occasion"]},
    # ── chunk 1: punctuation ───────────────────────────────────────────────
    {"id": "punct-1",  "description": "Missing comma after 'arrived'",   "keywords": ["comma", "arrived", "introductory"]},
    {"id": "punct-2",  "description": "Missing apostrophe: 'hadnt'",     "keywords": ["hadnt",  "hadn't"]},
    {"id": "punct-3",  "description": "Missing apostrophe: 'wasnt'",     "keywords": ["wasnt",  "wasn't"]},
    # ── chunk 2: whitespace ────────────────────────────────────────────────
    {"id": "ws-1",     "description": "Double space before 'archive'",   "keywords": ["archive",  "double space", "extra space", "whitespace"]},
    {"id": "ws-2",     "description": "Double space before 'perhaps'",   "keywords": ["perhaps",  "double space", "extra space", "whitespace"]},
    # ── chunk 2: em dash ───────────────────────────────────────────────────
    {"id": "em-1",     "description": "'--' after 'strategy'",           "keywords": ["strategy", "em dash", "double hyphen", "--"]},
    {"id": "em-2",     "description": "'--' after 'pen'",                "keywords": ["pen",      "em dash", "double hyphen"]},
    # ── chunk 3: repetition ────────────────────────────────────────────────
    {"id": "rep-1",    "description": "'walked' repeated four times",    "keywords": ["walked"]},
    # ── chunk 3: grammar ───────────────────────────────────────────────────
    {"id": "gram-1",   "description": "'arrived to' -> 'arrived at'",    "keywords": ["arrived to",    "arrived at"]},
    {"id": "gram-2",   "description": "'interested on' -> 'interested in'", "keywords": ["interested on", "interested in"]},
    {"id": "gram-3",   "description": "'capable to' -> 'capable of'",    "keywords": ["capable to",    "capable of"]},
    # ── chunk 5: hyphenation ───────────────────────────────────────────────
    {"id": "hyph-1",   "description": "'well known investigator' missing hyphen",  "keywords": ["well known", "well-known"]},
    {"id": "hyph-2",   "description": "'last minute decision' missing hyphen",     "keywords": ["last minute", "last-minute"]},
    {"id": "hyph-3",   "description": "'long established reputation' missing hyphen", "keywords": ["long established", "long-established"]},
    # ── chunk 5: number formatting ─────────────────────────────────────────
    {"id": "num-1",    "description": "'3' should be 'three'",  "keywords": ["3 notebooks", "three notebooks"]},
    {"id": "num-2",    "description": "'7' should be 'seven'",  "keywords": ["7 envelopes", "seven envelopes"]},
    {"id": "num-3",    "description": "'2' should be 'two'",    "keywords": ["2 keys", "two keys"]},
    # ── chunk 5: cuttable modifier ─────────────────────────────────────────
    {"id": "cut-1",  "description": "cuttable 'very' in 'very tired'",          "keywords": ["very tired"]},
    {"id": "cut-2",  "description": "cuttable 'really' in 'really late'",       "keywords": ["really late"]},
    # ── chunk 5: redundant that ────────────────────────────────────────────
    {"id": "rt-1",   "description": "redundant 'that' after 'noticed'",         "keywords": ["noticed that the lobby", "noticed that"]},
    {"id": "rt-2",   "description": "redundant 'that' after 'said'",            "keywords": ["said that she needed", "said that she"]},
    # ── chunk 5: weak verb ─────────────────────────────────────────────────
    {"id": "wv-1",   "description": "'walking slowly' → stronger single verb",  "keywords": ["walking slowly", "shuffled", "trudged", "plodded"]},
    {"id": "wv-2",   "description": "'looked at her carefully' → 'studied'",    "keywords": ["looked at her carefully", "studied", "scrutinised", "scrutinized"]},
    # ── chunk 6: wordy phrase ──────────────────────────────────────────────
    {"id": "wp-1",   "description": "'in order to review' is wordy",            "keywords": ["in order to review", "in order to"]},
    {"id": "wp-2",   "description": "'due to the fact that' is wordy",          "keywords": ["due to the fact that"]},
    # ── chunk 6: repetitive syntax ─────────────────────────────────────────
    {"id": "rs-1",   "description": "He moved / He arranged / He noted: same structure ×3", "keywords": ["he moved", "he arranged", "repetitive syntax", "repetitive structure", "same structure", "same opening"]},
    # ── chunk 6: vague word ────────────────────────────────────────────────
    {"id": "vw-1",   "description": "'the thing' is vague",                     "keywords": ["the thing", "vague word", "more specific"]},
    # ── chunk 6: unusual word ──────────────────────────────────────────────
    {"id": "uw-1",   "description": "'lacunose' is an unusual/obscure word",    "keywords": ["lacunose"]},
    # ── chunk 6: cliché metaphor ───────────────────────────────────────────
    {"id": "clm-1",  "description": "'silver lining' is a cliché metaphor",    "keywords": ["silver lining"]},
    # ── chunk 6: cliché adjective ──────────────────────────────────────────
    {"id": "cla-1",  "description": "'pitch black' is a cliché adjective",     "keywords": ["pitch black"]},
    {"id": "cla-2",  "description": "'burning desire' is a cliché adjective",  "keywords": ["burning desire"]},
    {"id": "cla-3",  "description": "'crystal clear' is a cliché adjective",   "keywords": ["crystal clear"]},
    # ── chunk 7: filter word ───────────────────────────────────────────────
    {"id": "fw-1",  "description": "filter word: 'noticed that the corridor'",  "keywords": ["noticed that the corridor", "noticed that"]},
    {"id": "fw-2",  "description": "filter word: 'saw that the window'",         "keywords": ["saw that the window", "saw that the"]},
    # ── chunk 7: throat-clearing ───────────────────────────────────────────
    {"id": "tc-1",  "description": "throat-clearing: 'It was only when...that she realised'", "keywords": ["it was only when", "throat-clear", "throat clear"]},
    {"id": "tc-2",  "description": "throat-clearing: 'What she found there was'",             "keywords": ["what she found there was", "what she found there"]},
    # ── chunk 7: passive voice ─────────────────────────────────────────────
    {"id": "pv-1",  "description": "passive voice: 'had been left...by the clerk'",  "keywords": ["by the clerk", "clerk left", "clerk had left"]},
    {"id": "pv-2",  "description": "passive voice: 'was handed...by an assistant'",  "keywords": ["by an assistant", "assistant handed", "handed her the envelope"]},
    # ── chunk 7: telling emotion ───────────────────────────────────────────
    {"id": "te-1",  "description": "telling emotion: 'felt frightened'",  "keywords": ["felt frightened", "frightened"]},
    {"id": "te-2",  "description": "telling emotion: 'felt relieved'",    "keywords": ["felt relieved", "relieved"]},

    # ══════════════════════════════════════════════════════════════════════
    # EXTENDED SET — S15-S28
    # ══════════════════════════════════════════════════════════════════════

    # ── S15: spelling ─────────────────────────────────────────────────────
    {"id": "spell-4",  "description": "'liason' -> 'liaison'",       "keywords": ["liason", "liaison"]},
    {"id": "spell-5",  "description": "'seperate' -> 'separate'",    "keywords": ["seperate", "separate"]},
    {"id": "spell-6",  "description": "'perseverence' -> 'perseverance'", "keywords": ["perseverence", "perseverance"]},
    # ── S16: punctuation ──────────────────────────────────────────────────
    {"id": "punct-4",  "description": "Missing comma after 'Once the room had cleared'", "keywords": ["once the room had cleared", "introductory", "comma"]},
    {"id": "punct-5",  "description": "Missing apostrophe: 'hadnt' (S16)",  "keywords": ["hadnt", "hadn't"]},
    {"id": "punct-6",  "description": "Missing apostrophe: 'couldnt'",      "keywords": ["couldnt", "couldn't"]},
    # ── S17: whitespace ───────────────────────────────────────────────────
    {"id": "ws-3",     "description": "Double space in 'The  filing system'",   "keywords": ["filing system", "double space", "extra space", "whitespace"]},
    {"id": "ws-4",     "description": "Double space in 'labels were  illegible'", "keywords": ["illegible", "double space", "extra space", "whitespace"]},
    # ── S17: em dash ──────────────────────────────────────────────────────
    {"id": "em-3",     "description": "'--' after 'principle'",   "keywords": ["principle", "em dash", "double hyphen"]},
    {"id": "em-4",     "description": "'--' after 'folder'",      "keywords": ["separate folder", "em dash", "double hyphen"]},
    # ── S18: repetition ───────────────────────────────────────────────────
    {"id": "rep-2",    "description": "'turned' repeated four times", "keywords": ["turned"]},
    # ── S19: grammar ──────────────────────────────────────────────────────
    {"id": "gram-4",   "description": "'differed to' -> 'differed from'",   "keywords": ["differed to", "differed from"]},
    {"id": "gram-5",   "description": "'comply to' -> 'comply with'",        "keywords": ["comply to", "comply with"]},
    {"id": "gram-6",   "description": "'responsible of' -> 'responsible for'", "keywords": ["responsible of", "responsible for"]},
    # ── S21: hyphenation ──────────────────────────────────────────────────
    {"id": "hyph-4",   "description": "'well timed intervention' missing hyphen",  "keywords": ["well timed", "well-timed"]},
    {"id": "hyph-5",   "description": "'high stakes agreement' missing hyphen",    "keywords": ["high stakes", "high-stakes"]},
    {"id": "hyph-6",   "description": "'long awaited process' missing hyphen",     "keywords": ["long awaited", "long-awaited"]},
    # ── S22: number formatting ─────────────────────────────────────────────
    {"id": "num-4",    "description": "'4' should be 'four'",   "keywords": ["4 folders", "four folders"]},
    {"id": "num-5",    "description": "'6' should be 'six'",    "keywords": ["6 pages", "six pages"]},
    {"id": "num-6",    "description": "'8' should be 'eight'",  "keywords": ["8 names", "eight names"]},
    # ── S23: cuttable modifier ─────────────────────────────────────────────
    {"id": "cut-3",  "description": "cuttable 'quite' in 'quite certain'",   "keywords": ["quite certain"]},
    {"id": "cut-4",  "description": "cuttable 'just' in 'just needed'",      "keywords": ["just needed"]},
    # ── S23: redundant that ────────────────────────────────────────────────
    {"id": "rt-3",   "description": "redundant 'that' after 'realized'",     "keywords": ["realized that the envelope", "realized that"]},
    {"id": "rt-4",   "description": "redundant 'that' after 'believed'",     "keywords": ["believed that the meeting", "believed that"]},
    # ── S23: weak verb ─────────────────────────────────────────────────────
    {"id": "wv-3",   "description": "'moved slowly towards' → stronger verb", "keywords": ["moved slowly", "shuffled", "crossed", "drifted"]},
    {"id": "wv-4",   "description": "'looked at her briefly' → 'glanced'",   "keywords": ["looked at her briefly", "glanced"]},
    # ── S24: wordy phrase ──────────────────────────────────────────────────
    {"id": "wp-3",   "description": "'at this point in time' is wordy",      "keywords": ["at this point in time"]},
    {"id": "wp-4",   "description": "'in the event that' is wordy",          "keywords": ["in the event that"]},
    # ── S24: vague word ────────────────────────────────────────────────────
    {"id": "vw-2",   "description": "'the stuff on the table' is vague",     "keywords": ["the stuff", "stuff on the table"]},
    # ── S24: repetitive syntax ─────────────────────────────────────────────
    {"id": "rs-2",   "description": "He cleared / He sorted / He photographed: same structure ×3", "keywords": ["he cleared", "he sorted", "he photographed", "repetitive syntax", "repetitive structure", "same opening"]},
    # ── S25: unusual word ──────────────────────────────────────────────────
    {"id": "uw-2",   "description": "'tenebrous' is an unusual/obscure word",  "keywords": ["tenebrous"]},
    # ── S25: cliché metaphor ───────────────────────────────────────────────
    {"id": "clm-2",  "description": "'heart of gold' is a cliché metaphor",   "keywords": ["heart of gold"]},
    # ── S25: cliché adjective ──────────────────────────────────────────────
    {"id": "cla-4",  "description": "'pitch black' (S25) is a cliché adjective",  "keywords": ["pitch black"]},
    {"id": "cla-5",  "description": "'burning desire' (S25) is a cliché adjective", "keywords": ["burning desire"]},
    {"id": "cla-6",  "description": "'blinding light' is a cliché adjective",       "keywords": ["blinding light"]},
    # ── S26: filter word ───────────────────────────────────────────────────
    {"id": "fw-3",  "description": "filter word: 'noticed that the briefcase'",  "keywords": ["noticed that the briefcase"]},
    {"id": "fw-4",  "description": "filter word: 'saw that the latch'",           "keywords": ["saw that the latch"]},
    # ── S26: throat-clearing ───────────────────────────────────────────────
    {"id": "tc-3",  "description": "throat-clearing: 'It was at that moment that she understood'", "keywords": ["it was at that moment", "that moment that she"]},
    {"id": "tc-4",  "description": "throat-clearing: 'What he found when he opened the drawer was'", "keywords": ["what he found when he opened", "what he found when"]},
    # ── S26: passive voice ─────────────────────────────────────────────────
    {"id": "pv-3",  "description": "passive: 'had been removed...by the deputy'",   "keywords": ["by the deputy", "deputy removed", "removed from the archive"]},
    {"id": "pv-4",  "description": "passive: 'was brought...by an assistant'",       "keywords": ["brought to her desk by", "assistant brought"]},
    # ── S26: telling emotion ───────────────────────────────────────────────
    {"id": "te-3",  "description": "telling emotion: 'felt anxious'",   "keywords": ["felt anxious", "anxious"]},
    {"id": "te-4",  "description": "telling emotion: 'was surprised'",  "keywords": ["was surprised", "surprised by what she read"]},

    # ── S28: inconsistent naming ───────────────────────────────────────────
    {"id": "naming-1", "description": "'Rennick' vs 'Renwick' in same passage",   "keywords": ["rennick", "renwick", "inconsistent"]},
    {"id": "naming-2", "description": "'Mireille' vs 'Mireil' in same passage",   "keywords": ["mireille", "mireil", "inconsistent"]},
    # ── S29: language (British spellings under American English style) ─────
    {"id": "lang-1",   "description": "'realised' should be 'realized'",           "keywords": ["realised", "realized", "british spelling", "spelling convention"]},
    {"id": "lang-2",   "description": "'organised' should be 'organized'",         "keywords": ["organised", "organized", "british spelling", "spelling convention"]},
    {"id": "lang-3",   "description": "'colour' should be 'color'",                "keywords": ["colour", "color", "british spelling", "spelling convention"]},
    {"id": "lang-4",   "description": "'recognised' should be 'recognized'",       "keywords": ["recognised", "recognized", "british spelling", "spelling convention"]},
]

CLEAN_CHUNKS = {4}  # Chunk 4 = S7; S27 chunk number TBD after run

# ---------------------------------------------------------------------------
# Document sections — one paragraph per section in the .docx file.
# ---------------------------------------------------------------------------
PARAGRAPHS = [
    # S1 — three spelling errors
    (
        "The hotel manager had offered to accomodate the full delegation in the east wing, a gesture "
        "Vera considered unnecessarily generous given the circumstances. She signed the register without "
        "looking up. The relavant documents had already been forwarded to her suite—three pages of "
        "correspondence, the amended contract, and a supplementary note from the legal team that no one "
        "had thought to summarize. She would attend to it all after dinner. For now, what mattered was "
        "that the occassion had passed without incident. The negotiations had stalled twice, but in the "
        "end both parties had initialled every page, and that was the only detail that would appear in "
        "tomorrow's briefing. She pocketed the carbon copy and took the stairs."
    ),
    # S2 — punctuation: missing comma after intro clause, hadnt, wasnt
    (
        "When the courier finally arrived she handed the envelope directly to Miriam without a word. "
        "The seal was intact, confirming the contents hadnt been examined since leaving the ministry. "
        "Miriam turned it over once before setting it beside her cold cup of tea. She had been expecting "
        "this for several days. The delay, she supposed, had been deliberate—a means of granting her "
        "time to prepare, or perhaps to force a decision she still wasnt sure she had the authority to "
        "make. She placed both palms flat on the table, considered the envelope for a long moment, then "
        "reached for it. She would read it. Of course she would; that had never really been in question. "
        "What she had been working through, these past days, was how she intended to respond. She broke the seal."
    ),
    # S3 — whitespace: double space in "The  archive" and "were  perhaps"
    (
        "The  archive room smelled of damp paper and something faintly chemical—fixative, or an old "
        "cleaning agent absorbed into the shelving over decades. Selby pulled the chain on the overhead "
        "lamp and waited for his eyes to adjust. The shelves ran floor to ceiling on three walls, dense "
        "with folders and cardboard boxes labeled in a cramped hand he could not read from the doorway. "
        "He had been told to find any series marked with a blue tab. There were  perhaps forty such "
        "folders visible from where he stood, and that was only the first shelf. He had not thought to "
        "bring a ladder. He rolled up his sleeves, noted the time, and started from the left. The lamp "
        "swung on its chain as he moved. He ignored it and kept working."
    ),
    # S4 — em dash: "--" used twice where "—" is document standard
    (
        "The inspector placed the photograph face down on the desk—he had found, he said, that the image "
        "itself was the distraction. \"We recovered it from the lower drawer,\" he said. \"Along with the "
        "correspondence.\" Hartmann said nothing. He had learned over many years that silence was the more "
        "effective strategy--it placed the burden on the other person, who invariably filled the space with "
        "more than they had intended to reveal. \"There are four letters—all dated within the same "
        "fortnight,\" the inspector said. He set the folder down. \"The postmarks don't match the return "
        "address.\" Hartmann set down his pen--a habit when he needed to think without appearing to—and "
        "looked at the folder. He would ask to see the letters before forming any opinion."
    ),
    # S5 — repetition: "walked" × 4
    (
        "She walked to the far window and looked down at the square below. A woman in a dark coat walked "
        "past the gate without pausing, arms folded against the cold. Vera watched until she walked out of "
        "sight around the corner of the building opposite, then let the curtain fall. The photographs "
        "remained spread across the low table where she had left them an hour ago, in the sequence Selby "
        "had suggested. She walked back to her chair and studied them without touching any of them. There "
        "were nine in total. The first three she had accounted for. The last two she was beginning to "
        "understand. The middle four she could not explain, and it was those that kept drawing her back. "
        "She reached for the one Selby had marked and held it under the lamp. The quality was poor, but "
        "the face was recognizable."
    ),
    # S6 — grammar: wrong prepositions
    (
        "Rennick arrived to the station shortly before noon, having been in transit since early morning. "
        "The journey had been long and uneventful, and he was stiff and short on patience by the time the "
        "train pulled in. The officer assigned to meet him seemed not to notice. \"Good timing,\" she said, "
        "moving quickly towards the exit. \"The team is very interested on this one.\" Rennick followed "
        "without answering. He had developed a practice, over years of this work, of forming no conclusions "
        "before seeing the scene directly. Whether he would prove capable to handle what waited for him "
        "depended entirely on what the scene had to show. He followed her through the barrier, into the "
        "main hall, and out into the street. Neither of them spoke until she unlocked the car."
    ),
    # S7 — clean: no errors (~2100 chars, guaranteed own chunk = chunk 4)
    (
        "The afternoon light had changed by the time Vera finished reading. She gathered the pages in order "
        "and returned them to the dossier, then carried it to the writing table by the window. Outside, the "
        "square had grown quieter. A few pigeons moved along the rim of the fountain. A man on one of the "
        "benches appeared to be asleep, his chin to his chest, one arm extended along the back of the seat. "
        "The linden trees at the far end were still in full leaf; it would be at least another month before "
        "any of that began to change.\n\n"
        "She poured a glass of water from the carafe on the sideboard and lingered at the window, "
        "looking out without focus. It was a habit she had carried from years ago—the "
        "practice of allowing her mind a few minutes of idleness after something demanding, before moving "
        "on. It served her well. She rarely needed to revisit a decision made this way.\n\n"
        "The dossier represented roughly eighteen months of work. Not all of it hers. A significant portion "
        "was compiled by the team in the capital, and some of the earlier documentation came through "
        "sources she could never trace to her satisfaction. Once, that uncertainty had nagged at her. "
        "It no longer did. You worked with the materials at hand, or the moment passed.\n\n"
        "Back at the writing table, she reached for the supplementary note from the legal team—three short "
        "paragraphs she had skimmed on the train without full attention. Reading carefully now, she found "
        "it confirmed what she had suspected since the opening meeting: the situation was manageable, "
        "provided the timing was correctly sequenced. That, at least, was ground she could work from.\n\n"
        "She wrote two words in the margin, underlined, closed it, and set it aside. The room had grown "
        "darker. She switched on the desk lamp, and the circle of light it cast made the space feel at "
        "once smaller and more settled. She still had half an hour before she was expected downstairs. "
        "She would use it."
    ),
    # S8 — hyphenation: three compound modifiers before nouns, missing hyphens
    (
        "The department had assigned a well known investigator to the case—a last minute decision that had "
        "surprised most of the team. He had a long established reputation for this kind of work, though his "
        "methods were sometimes considered unorthodox. No one objected. The file was handed to him that "
        "afternoon along with a brief summary of the known facts. He read both documents in his car before "
        "driving to the address. He had done this kind of work long enough to know that the summary was "
        "rarely the complete picture."
    ),
    # S9 — number formatting: 3, 7, and 2 should be spelled out
    (
        "The desk contained 3 notebooks, all filled in the same cramped hand, and 7 envelopes sorted by "
        "postmark. Rennick counted them twice to be sure. There were 2 keys on a short ring, a small "
        "compass, and a folded map of the district that had been annotated in red ink. He photographed "
        "everything before touching any of it. The notebooks he set aside for later. The envelopes he "
        "opened in order, reading each one in full before moving to the next."
    ),
    # S10 — cuttable modifiers, redundant that, weak verbs
    (
        "Mireille was very tired by the time she reached the hotel. It had been a long journey, and she "
        "had no desire to negotiate with anyone. She noticed that the lobby was empty—no porter in sight, "
        "no lamp burning at the front desk, only the low hum of the heating system and the faint sound of "
        "traffic from the street. She said that she needed a room for the night, walking slowly across the "
        "tiled floor toward reception. The clerk appeared from a back room and looked at her carefully from "
        "behind wire-rimmed glasses, taking in the coat, the mud on her shoes, the bag she clutched to her "
        "side. It was really late, and the street outside had fallen silent."
    ),
    # S11 — wordy phrases, repetitive syntax, vague word
    (
        "Rennick returned to the case file in order to review the evidence a second time. Due to the fact "
        "that the photographs were out of sequence, he could not determine the order of events with any "
        "certainty. He moved the folders to one side. He arranged the photographs in a single row across "
        "the desk. He noted the timestamp printed on the reverse of each one. The thing in the corner of "
        "the third photograph—an indistinct form he could not identify—continued to resist interpretation, "
        "and he found himself returning to it each time he worked through the sequence."
    ),
    # S12 — unusual word, cliché metaphor, cliché adjectives
    (
        "The corridor was pitch black when she stepped through the door, the overhead lights dead, the only "
        "source of illumination a grey rectangle of window at the far end. She had a burning desire to reach "
        "the archive room before anyone else, though her reasons for that urgency remained, even to herself, "
        "lacunose and difficult to name. His reports had always been crystal clear—precise to the point of "
        "austerity—but now they read as evasions. There was no silver lining in what she had uncovered. "
        "She pressed on regardless."
    ),
    # S13 — filter words, throat-clearing
    (
        "She noticed that the corridor had grown darker since she had last passed through it. She saw that "
        "the window at the far end was ajar, letting in a thin draught that moved along the floor. It was "
        "only when she reached the third door that she realized it was standing open. She stepped inside. "
        "What she found there was a single sheet of paper, left in the center of the desk with no "
        "indication of who had placed it or when."
    ),
    # S14 — passive voice, telling emotion
    (
        "The folder had been left on the desk by the clerk before he departed for the evening. She was "
        "handed the envelope by an assistant she did not recognize, a young woman who gave her name as "
        "Haase and said nothing further. She felt frightened when she opened it and read the contents: "
        "her name, her address, a date three days from now. She set the envelope down and looked at the "
        "door. She felt relieved only when she heard the lock click into place behind her."
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EXTENDED SET — S15-S28
    # ══════════════════════════════════════════════════════════════════════

    # S15 — spelling: liason, seperate, perseverence
    (
        "The liason officer had been stationed at the checkpoint for the better part of a year, long enough "
        "to know which vehicles required inspection and which could be waved through without comment. He had "
        "developed a practiced instinct for this kind of work. His task—and he was clear on this—was not to "
        "decide, but to seperate the categories: those that warranted attention and those that did not. The "
        "distinction was not always straightforward. What impressed his colleagues most was his perseverence "
        "under monotonous conditions; he brought the same quality of attention to the hundredth vehicle of "
        "the day as to the first, and his records were accordingly reliable."
    ),
    # S16 — punctuation: missing comma after intro clause, hadnt, couldnt
    (
        "Once the room had cleared Vera locked the door and stood for a moment with her back to it. The "
        "letter was still on the table. She hadnt read all of it—she had stopped at the third paragraph, "
        "where the figures were set out in a column, and had found she couldnt go further without sitting "
        "down first. She went to the chair by the window. From here she could see the far edge of the "
        "square, the bench where she sometimes ate her lunch, the linden tree that had been there since "
        "before she was born. She read the rest of the letter from there, slowly, and when she was done "
        "she folded it once and placed it in the inside pocket of her coat."
    ),
    # S17 — whitespace ×2, em dash ×2
    (
        "The  filing system had not been updated in over three years, a fact that became apparent as soon "
        "as Brennan began working through the first drawer. The labels were  illegible in places, the "
        "folders misaligned, and several documents had been filed under categories that no longer existed "
        "in any current taxonomy. He worked steadily, making no comment. The task was straightforward in "
        "principle--he was simply finding and extracting specific records--but the volume of material was "
        "greater than he had been led to expect. He set aside anything dated after the reorganisation. By "
        "the end of the afternoon he had a stack of some forty pages. He placed these in a separate "
        "folder--one he had brought for exactly this purpose--and locked it in his briefcase."
    ),
    # S18 — repetition: "turned" × 4
    (
        "She turned to face the corridor. At the far end, a figure turned and walked away before she could "
        "identify him. She stood for a moment, then turned back towards the door. The folder was where she "
        "had left it, undisturbed on the edge of the desk. She picked it up and turned it over in her hands "
        "before opening it. The photograph inside was new. She had not placed it there. She set the folder "
        "down and looked again at the corridor, which was now empty and gave nothing away."
    ),
    # S19 — grammar: differed to, comply to, responsible of
    (
        "Rennick's approach to these matters differed to the standard procedure in several respects that his "
        "supervisors had noted but not formally addressed. He was not, technically speaking, in violation of "
        "any regulation; he had simply developed a method of working that did not comply to the department's "
        "preferred protocols. No one had yet decided who was responsible of enforcing those protocols when "
        "the person in question produced results. That ambiguity had protected him thus far, and he was not "
        "unaware of it. He intended to keep it that way for as long as the arrangement held."
    ),
    # S20 — CLEAN
    (
        "The morning light came through the high windows at a low angle, laying long rectangles across the "
        "floor of the reading room. Hartmann worked at the table near the door, the only surface large "
        "enough to spread the documents without overlap. He had arranged them chronologically, left to "
        "right, which was not the order in which they had been found, but the order in which they made "
        "sense. He worked without speaking. His assistant sat across the room, taking notes in a shorthand "
        "that Hartmann had never been able to read.\n\n"
        "By mid-morning they had established the sequence of events up to the point of the second "
        "transaction. Beyond that, the record became fragmentary. There were gaps in the correspondence, "
        "periods of three or four weeks during which nothing appeared to have been committed to paper. "
        "Whether this represented a genuine absence of activity or simply a different mode of record-keeping "
        "was not yet clear. Hartmann noted the gaps on a separate sheet and set it aside."
    ),
    # S21 — hyphenation: well timed, high stakes, long awaited
    (
        "It had been a well timed intervention: the department had moved before the window closed, and the "
        "resulting settlement, while imperfect, was a high stakes agreement that neither side had expected "
        "to reach in a single session. The team's satisfaction was tempered by the knowledge that what lay "
        "ahead was harder work—the long awaited process of implementation, which historically was where "
        "these agreements became undone. Svensson filed the preliminary documents and returned to his desk. "
        "He would review the final language in the morning, when he was better placed to assess it."
    ),
    # S22 — number: 4, 6, 8 should be spelled out
    (
        "The desk held 4 folders, each marked with a date in the same cramped handwriting, and a glass "
        "paperweight that served no obvious purpose. Brennan counted the folders twice. In the lower drawer "
        "he found 6 pages of correspondence, all addressed to the same recipient and none of them signed. "
        "Pinned to the inside of the drawer was a note in a different hand, listing 8 names in two columns "
        "with a date written beside each one. He photographed everything before touching any of it. The "
        "folders he read in order. The correspondence he set aside for later."
    ),
    # S23 — cuttable: quite, just; redundant that: realized, believed; weak verb ×2
    (
        "Miriam was quite certain by that point that the meeting had been called for reasons that had not "
        "been disclosed to her. She just needed a few more minutes to confirm what she already suspected. "
        "She realized that the envelope she had been given was not addressed to her name but to a name she "
        "did not recognize. She believed that the meeting had been arranged at short notice, which explained "
        "the absence of the usual documentation. She moved slowly towards the door, watching the others as "
        "she went. The chairman looked at her briefly from across the table before turning back to his papers."
    ),
    # S24 — wordy: at this point in time, in the event that; vague: stuff; repetitive syntax ×3
    (
        "At this point in time, the only reliable account of what had occurred came from a single witness "
        "whose credibility had not yet been established. In the event that a second source could be "
        "confirmed, the investigation would stand on firmer ground. The stuff on the table—a loose "
        "collection of objects Rennick could not immediately account for—suggested a degree of preparation "
        "he had not anticipated. He cleared the surface. He sorted the items into two groups. He "
        "photographed each group in turn before replacing anything."
    ),
    # S25 — unusual: tenebrous; cliché metaphor: heart of gold; cliché adj: pitch black, burning desire, blinding light
    (
        "The cellar was pitch black when she reached the bottom of the stairs, the single bulb overhead "
        "long dead. She had a burning desire to find the files before anyone else arrived, though she could "
        "not have explained why urgency felt so appropriate to a task that was, technically, entirely "
        "routine. Selby, for all his faults, had a heart of gold—he would not have withheld the information "
        "deliberately. The beam of her torch produced a blinding light that swept the shelves in slow arcs. "
        "She moved forward into the tenebrous space beyond the stairs, where the shelving ran deeper than "
        "she had expected and the air tasted of old paper and something faintly chemical."
    ),
    # S26 — filter word ×2, throat-clearing ×2, passive ×2, telling emotion ×2
    (
        "She noticed that the briefcase had been moved since she last entered the room. She saw that the "
        "latch was now closed, where before it had been left open. It was at that moment that she understood "
        "what had happened in her absence. What he found when he opened the drawer was a photograph of "
        "himself, taken from a distance without his knowledge. The documents had been removed from the "
        "archive by the deputy before his departure that morning. The envelope was brought to her desk by "
        "an assistant she did not recognize. She felt anxious as she opened it, her eyes moving quickly to "
        "the signature at the bottom of the page. She was surprised by what she read there."
    ),
    # S28 — inconsistent naming: Rennick/Renwick, Mireille/Mireil
    (
        "Rennick arrived at the address twelve minutes early and spent the time reviewing notes he had made "
        "that morning. The building was not what he had expected: a narrow converted house on a residential "
        "street, no signage, the windows frosted. Renwick checked the number against the paper he had been "
        "given. It matched. He rang the bell. Mireille opened the door herself—he had expected an "
        "intermediary of some kind—and stepped back without greeting him. He followed her inside. Mireil "
        "had already laid the documents on the table. She left the room while he read them."
    ),
    # S29 — language: British spellings flagged under American English house style
    (
        "She realised as she reached the landing that the building had an entirely different character after "
        "dark. What she had assumed to be organised confusion during the working day resolved, in the quiet, "
        "into something more deliberate. The colour of the walls—a pale grey she had not remarked on "
        "before—seemed to absorb what little light remained. She recognised the door at the far end from "
        "the photograph Selby had shown her, but had not registered, until now, that it was fitted with a "
        "second lock. She tried the handle. It was open."
    ),
    # S27 — CLEAN
    (
        "The train left the station twelve minutes late, which was not unusual for that line at that hour. "
        "Svensson found his compartment, stowed his bag in the rack above the seat, and settled by the "
        "window. He had a two-hour journey ahead of him and had intended to spend it reviewing the notes "
        "from the morning's session, but within twenty minutes of departure he found himself looking out at "
        "the landscape instead.\n\n"
        "The fields at that time of year were flat and pale, cut back after harvest, the trees along the "
        "margins already beginning to thin. There was a quality to the light in the late afternoon that he "
        "had always associated with transition—not autumn exactly, not the end of things, but the period "
        "just before the end, when the change had become inevitable but had not yet fully arrived. He had "
        "no particular feelings about it. It was simply the landscape, doing what it did at that time of "
        "year, indifferent to the people moving through it."
    ),
]

CLEAN_SECTION_IDX = 6   # S7 (0-indexed) — original clean section
CLEAN_SECTION_IDX2 = 19  # S20 (0-indexed) — second clean section
CLEAN_SECTION_IDX3 = 26  # S27 (0-indexed) — third clean section


def create_test_doc() -> None:
    doc = Document()
    for para in PARAGRAPHS:
        doc.add_paragraph(para)
    doc.save(TEST_DOC_PATH)
    print(f"Test document written: {TEST_DOC_PATH}")


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
    create_test_doc()
    print("Running reviewer on dev set...")
    results, usage = review_document(str(TEST_DOC_PATH), on_progress=_progress)
    print()

    if results:
        print("RAW OUTPUT\n" + "-" * 62)
        for r in results:
            print(f"\n-- Chunk {r.chunk}/{r.total} --\n   {r.preview}")
            for f in r.flags:
                print(f"  [{f.tier.upper()}] {f.type}")
                print(f"    TEXT:  {f.text}")
                print(f"    ISSUE: {f.issue}")

    score(results, usage)


if __name__ == "__main__":
    main()
