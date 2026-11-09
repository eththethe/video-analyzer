# video-analyzer

Judges Sahiyya+ ad videos with AI models and turns the results into a Meta ad-set grouping and deployment plan.

## Latest versions (as of 9 Nov 2026)

The scripts pick the newest prompt of each kind on their own. File names sort by date (`YYMMDD`), then by letter. No setting is needed.

| Kind | Latest file | Used by | Status |
|---|---|---|---|
| Scoring | [`260929a_prompt_scoring.md`](prompts/260929a_prompt_scoring.md) | `gemini.py`, `text_judge.py` | Current. Used for every scoring run since 29 Sep, including the 4 Oct runs (`superstar-ai`, `wp2026-jan-mar`) |
| Grouping | [`261004b_prompt_grouping.md`](prompts/261004b_prompt_grouping.md) | `group.py` | Current. Used for the 4 Oct plan `superstar-wp-janmar` (ad sets of 5, organised by disease, built on the Hook & CTA Playbook) |
| Context | [`260927a_prompt_context.md`](prompts/260927a_prompt_context.md) | inserted into the scoring and grouping prompts at `{context}` | Current. No newer context file exists, so 260929a and 261004b both use it |
| Laya questions | [`260929a_prompt_laya.json`](prompts/260929a_prompt_laya.json) | `laya_judge.py` | Current |

Models:

- **Gemini panel:** `gemini-3.8-flash` and `gemini-3.1-pro-preview` (`PANEL_MODELS` in `.env`). Both models judge every video.
- **Grouping plan:** written by the last panel model (`gemini-3.1-pro-preview`) unless `--model` is given.
- **Text judges:** GPT Sol on Azure (`SOL_MODEL_NAME`), Claude Sonnet through the local `claude` CLI, and Laya (`laya-multilingual`).

Older prompt versions are kept as a record of what earlier runs used. See [prompts/README.md](prompts/README.md) for the full version history and for how to add a new version.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `judging/upload.py` | Uploads `vid/*.mp4` to the GCS bucket once, so judging runs can reuse the copies |
| 2 | `judging/gemini.py` | Each Gemini panel model watches each video, records evidence (shot log, transcript, claims, CTAs), and scores it 0-100 |
| 3 | `judging/text_judge.py`, `judging/laya_judge.py` | Optional second opinions from text-only models, based on Gemini's written evidence. Metrics that need sight or sound are left out, so these scores are marked provisional |
| 4 | `judging/group.py` | Groups the judged videos into ad sets and writes the deployment plan |
| 5 | `judging/combine.py` | Combines the newest scoring and grouping CSVs of every model into one Excel file |
| - | `judging/meta_ads.py` | Exports lifetime Meta ad performance for every ad account the token can access |

Every result records the `prompt_file` and a `prompt_version` fingerprint, so each row can be traced to the exact prompt that produced it.

## Setup

```
pip install google-genai google-cloud-storage pydantic pandas openpyxl openai requests httpx python-dotenv
gcloud auth application-default login
```

Put keys and settings in `.env` (it is gitignored): `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=global`, `GCS_BUCKET`, `PANEL_MODELS`, `META_ACCESS_TOKEN`, `AZURE_ENDPOINT`, `AZURE_API_KEY`, `API_VERSION`, `SOL_MODEL_NAME`, `IMPOSSIBL_API_KEY`. The docstring at the top of each script lists what that script needs.

Not in git: source videos (`vid/`, about 12 GB) and all generated output (`judging/results/`).

## Changing a prompt

Don't edit an old prompt file. Copy the newest one to a new name with today's date, for example `261110a_prompt_grouping.md` (`261110b_...` for a second version on the same day), and edit the copy. The scripts will use it automatically. Then add a row for it to the table above and to [prompts/README.md](prompts/README.md).

If the new prompt changes what the model must return, update the output schema in the script too: `Observation` / `Scores` / `Assessment` in `gemini.py`, and `BatchRow` / `Families` / `Plan` in `group.py`.
