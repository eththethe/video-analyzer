# Prompts

The judging scripts read these files at run time, so edits here take effect on the next run.

| File | Used by | Placeholders |
|---|---|---|
| `scoring_prompt.md` | `judging/gemini.py` (system prompt for every video) | `{context}` |
| `grouping_prompt.md` | `judging/group.py` (creative grouping and deployment plan) | `{context}`, `{n}`, `{cards}` |
| `context.md` | inserted into both prompts at `{context}` | none |

Put account facts that both prompts need (audience, product facts, what Jeanette's team values) in `context.md` so they stay in sync.

Scoring prompt notes (from the original file, version 2026-09-27): personalized for Jeanette's team. Weights and decision thresholds are proposed internal rubric settings, not empirically validated performance predictors. A product playbook, approved claims, campaign objective, destination, benchmark creatives and performance data can optionally be supplied.

If you change what either prompt asks the model to return (new metrics, new label values, new output sections), the output structure in the script has to change too: `Observation` / `Scores` / `Assessment` in `gemini.py`, `Classification` / `Plan` in `group.py`.
