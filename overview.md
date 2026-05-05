# Codebase Overview

## core/reviewer.py — Copyedit engine & API plumbing

| Name | Purpose | Related |
|------|---------|---------|
| `Flag` | Dataclass for a single copyedit flag (type, quote, comment) | `ChunkResult`, `parse_flags` |
| `TokenUsage` | Tracks input/output token counts per API call | `ChunkResult`, `review_document` |
| `ChunkResult` | Bundles flags + token usage for one text chunk | `Flag`, `review_document` |
| `_get_api_key()` | Reads Anthropic key from env/keyring | `_get_async_client`, `review_document` |
| `_get_async_client()` | Returns a cached AsyncAnthropic client | `_get_api_key`, `review_document` |
| `load_docx()` | Extracts plain text from a .docx file | `chunk_text`, `review_document` |
| `chunk_text()` | Splits text into sentence-boundary chunks for the API | `load_docx`, `review_document` |
| `parse_flags()` | Parses Claude's XML response into `Flag` objects | `Flag`, `review_document` |
| `review_document()` | Orchestrates async chunk-by-chunk copyedit review | `parse_flags`, `chunk_text` |

---

## core/prompts.py — System prompt assembly

| Name | Purpose | Related |
|------|---------|---------|
| `_load_error_types()` | Loads error-type definitions from JSON | `_build_error_rule`, `load_error_types` |
| `_build_error_rule()` | Formats one error type into a prompt rule string | `_load_error_types`, `build_system_prompt` |
| `_load_style_config()` | Loads user style settings from JSON | `_build_style_block`, `load_style_config` |
| `_build_style_block()` | Renders style settings as a prompt section | `_load_style_config`, `build_system_prompt` |
| `load_error_types()` | Public accessor for error-type list | `_load_error_types`, `get_valid_types` |
| `load_style_config()` | Public accessor for style config dict | `_load_style_config`, `save_style_config` |
| `save_style_config()` | Persists updated style settings to JSON | `load_style_config`, `_load_style_config` |
| `get_valid_types()` | Returns set of enabled error-type IDs | `load_error_types`, `build_system_prompt` |
| `build_system_prompt()` | Assembles full system prompt from rules + style | `get_valid_types`, `_build_style_block` |
| `build_review_prompt()` | Wraps a text chunk in the user-turn prompt | `build_system_prompt`, `review_document` |

---

## core/cache.py — Disk caching for all analyzers

| Name | Purpose | Related |
|------|---------|---------|
| `_file_hash()` / `_file_hash_cached()` | SHA-256 of source file for cache keying | `_cache_path`, `save_*` |
| `_cache_path()` | Returns Path for a given file + analysis kind | `_file_hash`, all `save_*`/`load_*` |
| `_flag_to_dict()` / `_flag_from_dict()` | Serialize/deserialize `Flag` | `save_copyedit_cache`, `load_copyedit_cache` |
| `_chunk_to_dict()` / `_chunk_from_dict()` | Serialize/deserialize `ChunkResult` | `save_copyedit_cache`, `load_copyedit_cache` |
| `save_copyedit_cache()` / `load_copyedit_cache()` | Persist/restore copyedit results | `_chunk_to_dict`, `_cache_path` |
| `_sheet_to_dict()` / `_sheet_from_dict()` | Serialize/deserialize `CharacterSheet` | `save_arch_cache`, `save_emotion_cache` |
| `save_arch_cache()` / `load_arch_cache()` | Persist/restore character archetype results | `_sheet_to_dict`, `_cache_path` |
| `save_emotion_cache()` / `load_emotion_cache()` | Persist/restore emotion analysis results | `_sheet_to_dict`, `_cache_path` |
| `_genre_result_to_dict()` / `_genre_result_from_dict()` | Serialize/deserialize `GenreResult` | `save_genre_cache`, `load_genre_cache` |
| `save_synopsis_cache()` / `load_synopsis_cache()` | Persist/restore synopsis/plot results | `_cache_path`, `analyze_synopsis` |
| `save_genre_cache()` / `load_genre_cache()` | Persist/restore genre detection results | `_genre_result_to_dict`, `_cache_path` |
| `save_opening_cache()` / `load_opening_cache()` | Persist/restore opening analysis results | `_cache_path`, `analyze_opening` |
| `save_arcs_cache()` / `load_arcs_cache()` | Persist/restore structural arc results | `_cache_path`, `analyze_arcs` |

---

## core/character_analyzer.py — Character archetype analysis

| Name | Purpose | Related |
|------|---------|---------|
| `_chunk()` | Splits text into fixed-size character-aware chunks | `analyze_characters`, `_annotation_system` |
| `load_archetypes()` | Loads archetype definitions from JSON | `_annotation_system`, `analyze_characters` |
| `RosterEntry` | Represents one discovered character (name, aliases) | `_parse_discovery`, `_build_sheets` |
| `_Signal` | One archetype/emotion signal event for a character | `_parse_signals`, `_build_sheets` |
| `CharacterSheet` | Aggregated archetype profile for a character | `_build_sheets`, `save_arch_cache` |
| `_discovery_prompt()` | Prompt asking Claude to find characters in a chunk | `_parse_discovery`, `analyze_characters` |
| `_parse_discovery()` | Parses Claude's character-list response into `RosterEntry` list | `_discovery_prompt`, `analyze_characters` |
| `_parse_reconciliation()` | Merges/deduplicates character names across chunks | `_parse_discovery`, `analyze_characters` |
| `_first_word()` | Extracts first name token for alias matching | `_parse_reconciliation`, `_parse_discovery` |
| `_annotation_system()` | Builds system prompt for archetype signal annotation | `_parse_signals`, `analyze_characters` |
| `_annotation_prompt()` | User-turn prompt for one annotation chunk | `_annotation_system`, `analyze_characters` |
| `_parse_signals()` | Parses archetype signal XML from Claude into `_Signal` list | `_build_sheets`, `_annotation_system` |
| `_build_sheets()` | Aggregates signals into per-character `CharacterSheet` objects | `_parse_signals`, `analyze_characters` |
| `_negative_annotation_system()` | System prompt for a second pass that suppresses false positives | `_merge_neg_signals`, `analyze_characters` |
| `_merge_neg_signals()` | Applies negative-pass results to remove false-positive signals | `_negative_annotation_system`, `_build_sheets` |
| `analyze_characters()` | Top-level entry point: full character archetype pipeline | `_build_sheets`, `save_arch_cache` |

---

## core/emotion_analyzer.py — Character emotion arc analysis

| Name | Purpose | Related |
|------|---------|---------|
| `load_emotions()` | Loads emotion-category definitions from JSON | `_emotion_annotation_system`, `analyze_emotions` |
| `_emotion_annotation_system()` | System prompt for emotion signal annotation | `_parse_emotion_signals`, `analyze_emotions` |
| `_parse_emotion_signals()` | Parses emotion signal XML into `_Signal` list | `_build_emotion_sheets`, `_emotion_annotation_system` |
| `_build_emotion_sheets()` | Aggregates emotion signals into `CharacterSheet` objects | `_parse_emotion_signals`, `analyze_emotions` |
| `_suppression_annotation_system()` | System prompt for false-positive suppression pass | `_merge_suppression_signals`, `analyze_emotions` |
| `_merge_suppression_signals()` | Removes suppressed emotion signals from sheets | `_suppression_annotation_system`, `_build_emotion_sheets` |
| `analyze_emotions()` | Top-level entry point: full emotion arc pipeline | `_build_emotion_sheets`, `save_emotion_cache` |

---

## core/genre_analyzer.py — Genre & trope detection

| Name | Purpose | Related |
|------|---------|---------|
| `load_genres()` | Loads genre/trope definitions from JSON | `_annotation_system`, `analyze_genres` |
| `TropeSignal` | One detected trope instance (genre, quote, location) | `_parse_signals`, `_build_results` |
| `GenreResult` | Aggregated genre score with supporting trope signals | `_build_results`, `save_genre_cache` |
| `_annotation_system()` | System prompt for trope detection | `_parse_signals`, `analyze_genres` |
| `_parse_signals()` | Parses trope signal XML into `TropeSignal` list | `_deduplicate`, `_annotation_system` |
| `_token_overlap()` | Jaccard-style token overlap for deduplication | `_deduplicate`, `_parse_signals` |
| `_deduplicate()` | Removes near-duplicate trope signals | `_build_results`, `_parse_signals` |
| `_build_results()` | Aggregates tropes into scored `GenreResult` list | `_deduplicate`, `analyze_genres` |
| `analyze_genres()` | Top-level entry point: full genre detection pipeline | `_build_results`, `save_genre_cache` |

---

## core/synopsis_analyzer.py — Plot structure & synopsis

| Name | Purpose | Related |
|------|---------|---------|
| `PlotBeat` | One labeled story beat (name, position, description) | `_parse_beat_labels`, `SynopsisResult` |
| `SynopsisResult` | Full plot analysis: synopsis, themes, beats, arcs | `_parse_synthesis`, `analyze_synopsis` |
| `_parse_synthesis()` | Parses Claude's synthesis response into `SynopsisResult` | `SynopsisResult`, `analyze_synopsis` |
| `_parse_beat_labels()` | Parses beat-labeling response into `PlotBeat` list | `PlotBeat`, `analyze_synopsis` |
| `analyze_synopsis()` | Top-level entry point: synopsis + beat labeling pipeline | `_parse_synthesis`, `save_synopsis_cache` |

---

## core/arc_analyzer.py — Structural story arc analysis

| Name | Purpose | Related |
|------|---------|---------|
| `load_structural_arcs()` | Loads structural arc templates from JSON | `_build_context`, `analyze_arcs` |
| `ArcResult` | Score + evidence for one structural arc | `_parse_arcs`, `ArcsResult` |
| `ArcsResult` | Collection of all arc results for the manuscript | `analyze_arcs`, `save_arcs_cache` |
| `_build_context()` | Builds synopsis context string from `SynopsisResult` | `analyze_arcs`, `ArcResult` |
| `_parse_arcs()` | Parses Claude's arc-scoring response into `ArcResult` list | `ArcResult`, `analyze_arcs` |
| `analyze_arcs()` | Top-level entry point: arc scoring against synopsis | `_parse_arcs`, `save_arcs_cache` |

---

## core/opening_analyzer.py — Opening pages evaluation

| Name | Purpose | Related |
|------|---------|---------|
| `OpeningCheckItem` | One checklist item result (name, score, notes) | `OpeningResult`, `_parse_result` |
| `OpeningResult` | Full opening evaluation: items + overall verdict | `_parse_result`, `analyze_opening` |
| `_first_words()` | Extracts the first N words from manuscript text | `analyze_opening`, `_parse_result` |
| `_parse_result()` | Parses Claude's opening-check response into `OpeningResult` | `OpeningResult`, `analyze_opening` |
| `analyze_opening()` | Top-level entry point: opening pages analysis | `_parse_result`, `save_opening_cache` |

---

## core/docx_comments.py — Inject Word comments into .docx

| Name | Purpose | Related |
|------|---------|---------|
| `_norm()` / `_clean_needle()` | Normalize/clean text for fuzzy span matching | `_find_span`, `_anchor_comment` |
| `_find_span()` | Locates a flag's quoted text within a paragraph | `_anchor_comment`, `_sig_words` |
| `_sig_words()` | Extracts significant (non-stopword) words for fallback search | `_find_span`, `_anchor_comment` |
| `_q()` | Prefixes a tag name with the Word XML namespace | `_make_comment_xml`, `_comment_range_start` |
| `_make_comment_xml()` | Builds the `comments.xml` part from all comment data | `annotate`, `_q` |
| `_comment_range_start/End()` | Creates `<w:commentRangeStart/End>` XML elements | `_insert_markers_around`, `_make_comment_xml` |
| `_comment_ref_run()` | Creates `<w:commentReference>` run element | `_insert_markers_around`, `_comment_range_end` |
| `_para_runs()` | Returns all `<w:r>` run elements within a paragraph | `_split_run`, `_anchor_comment` |
| `_run_text()` / `_set_run_text()` | Get/set text content of a run element | `_split_run`, `_anchor_comment` |
| `_split_run()` | Splits a run at a character offset to allow mid-run anchoring | `_insert_markers_around`, `_anchor_comment` |
| `_insert_markers_around()` | Wraps a span of runs with comment range markers | `_anchor_comment`, `_split_run` |
| `_anchor_comment()` | Finds the right paragraph and anchors one comment in the XML | `annotate`, `_insert_markers_around` |
| `annotate()` | Main entry point: writes annotated copy of the .docx | `_anchor_comment`, `_make_comment_xml` |

---

## main.py — GUI application (tkinter)

| Name | Purpose | Related |
|------|---------|---------|
| `_fmt_cache_dt()` | Formats a cached ISO timestamp for display | `App`, `load_*_cache` |
| `_softmax_sizes()` | Maps archetype scores to font sizes via softmax | `_build_arch_fig`, `_render_archetype_plot` |
| `_flat_btn()` | Creates a styled flat tkinter button | `App`, `_cog_btn` |
| `_draw_arch_bars()` | Draws archetype bar chart on a canvas | `_build_arch_fig`, `_make_arch_plot_widget` |
| `_build_arch_fig()` | Builds matplotlib figure for archetype word-cloud/bars | `_render_archetype_plot`, `_make_arch_plot_widget` |
| `_build_emotion_fig()` | Builds matplotlib figure for emotion arc lines | `_render_emotion_plot`, `_make_emotion_plot_widget` |
| `_render_archetype_plot()` | Renders archetype figure to a PNG bytes buffer | `_build_arch_fig`, `_make_arch_plot_widget` |
| `_make_arch_plot_widget()` | Creates tkinter widget embedding the archetype plot | `_render_archetype_plot`, `App` |
| `_render_emotion_plot()` | Renders emotion figure to a PNG bytes buffer | `_build_emotion_fig`, `_make_emotion_plot_widget` |
| `_make_emotion_plot_widget()` | Creates tkinter widget embedding the emotion plot | `_render_emotion_plot`, `App` |
| `_make_emotion_radar_widget()` | Creates radar chart widget for per-character emotion breakdown | `_make_emotion_plot_widget`, `App` |
| `_make_genre_pie_widget()` | Creates pie chart widget for genre distribution | `_make_genre_plot_widget`, `App` |
| `_build_genre_fig()` | Builds matplotlib figure for genre bar chart | `_make_genre_plot_widget`, `_make_genre_pie_widget` |
| `_make_genre_plot_widget()` | Creates tkinter widget embedding the genre bar chart | `_build_genre_fig`, `App` |
| `_write_arch_report()` | Writes archetype analysis to a .docx report file | `App`, `_make_arch_plot_widget` |
| `_write_emotion_report()` | Writes emotion analysis to a .docx report file | `App`, `_make_emotion_plot_widget` |
| `_write_genre_report()` | Writes genre analysis to a .docx report file | `App`, `_make_genre_plot_widget` |
| `_cog_btn()` | Creates a small gear-icon settings button | `_flat_btn`, `App` |
| `_write_opening_report()` | Writes opening analysis to a .docx report file | `App`, `analyze_opening` |
| `_write_plot_report()` | Writes synopsis + arc analysis to a .docx report file | `App`, `analyze_synopsis` |
| `App` | Main tkinter window: file picker, tabs, analysis triggers, progress | all `analyze_*`, all `_write_*_report` |
