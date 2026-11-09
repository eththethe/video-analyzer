# Prompts

Every prompt version is its own file, named `<YYMMDD><letter>_prompt_<kind>.md`:

| Kind | Used by | Placeholders |
|---|---|---|
| `scoring` | `judging/gemini.py` (system prompt for every video) | `{context}` |
| `grouping` | `judging/group.py` (creative grouping and deployment plan) | `{context}`, `{n}`, `{cards}` |
| `context` | inserted into the prompts at `{context}` | none |
| `laya` (`.json`) | `judging/laya_judge.py` (questions, levels and Laya model) | none |

The scripts always use the **newest** file of each kind (file names sort by date, then letter). `{context}` is filled from the newest context file dated at or before the prompt itself.

**To change a prompt, don't edit an old file.** Copy the newest one to a new name, for example `260930a_prompt_scoring.md` (a second version on the same day would be `260930b_...`), and edit the copy. Old files stay as the record of what earlier runs used; every result records its `prompt_file`, and the combined Excel shows it next to `source_file`.

| Version | What it is |
|---|---|
| `260926a_prompt_scoring.md` | The rubric used for the 26 Sep run (19 metrics, 0-100, Virlo-style categories) |
| `260926a_prompt_grouping.md` | The 26 Sep grouping prompt: concepts and near-duplicates only; code built the ad sets |
| `260927a_prompt_scoring.md` | Jeanette's team scoring prompt (27 metrics, release status, claim audit), used from 27 Sep |
| `260927a_prompt_grouping.md` | Jeanette's team grouping and deployment-plan prompt, used from 28 Sep |
| `260927a_prompt_context.md` | Account context shared by the 260927a prompts (and 260929a, which has no context file of its own) |
| `260929a_prompt_scoring.md` | 260927a plus a `short_summary` (max 25 words, for media buyers), written as the last column of the scoring CSV |
| `260929a_prompt_laya.json` | Laya questions: each text-judgeable metric of the scoring rubric asked as a 5-level question |
| `261004a_prompt_grouping.md` | Grouping built on the Sahiyya+ Hook & CTA Playbook (204 Facebook ads with ROAS): ad sets of exactly 5 ads, at least 5 ad sets, 1 TOFU / 2 MOFU / 2 BOFU per ad set, organised by disease; classifies each video's playbook hook, close and video type; release status, compliance flags and brand-safety tier are not used (left out of the cards). Uses the 260927a context. Never completed a run (replaced the same evening) |
| `261004b_prompt_grouping.md` | 261004a, but a video may be used in more than one ad set (still never twice in the same ad set); the ledger lists every ad set a LAUNCH video is in |

Scoring prompt notes (from the original file, version 2026-09-27): personalized for Jeanette's team. Weights and decision thresholds are proposed internal rubric settings, not empirically validated performance predictors.

The 26 Sep prompts were also guided by the field descriptions in the output structure of the scripts at that time, and the user message per video was "Break this video down, describe it in detail, then judge it using the rubric."

If a new prompt changes what the model must return (new metrics, new label values, new output sections), the output structure in the script has to change too: `Observation` / `Scores` / `Assessment` in `gemini.py`, `BatchRow` / `Families` / `Plan` in `group.py`.
