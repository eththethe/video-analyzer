"""Combine the latest scoring and grouping CSVs of every model into one Excel file.

Output: judging/results/result_combined/sahiyya_analysis[_run-<stamp>]_<YYYYmmdd_HHMMSS>.xlsx
    Scoring tab   every model's scoring rows stacked
    Grouping tab  every model's grouping rows stacked
Both tabs start with file, run_id (e.g. scoring_0927_2315 = the scoring run
started 27 Sep 23:15), source_file (the CSV), prompt_file (the prompts/*.md that
produced the row), then llm_model beside overall_score (called creative_score or
overall in the CSVs), so every row can be traced back to its run, CSV and prompt.
prompt_file comes from the CSV when it records one, otherwise it is the newest
prompt version dated on or before the run. All other columns, including the
short_summary the scoring model writes (last column), come straight from the CSVs.

By default each model's newest untagged file in result_scoring/ and
result_grouping/ is used. --stamp picks an older run instead: files whose
timestamp starts with it. --tag uses the files of a tagged video set instead
(gemini.py/group.py --gcs ... --tag, e.g. results_<model>_<stamp>_superstar-ai.csv).
--tag can be repeated: each tag's newest files are stacked, with a video_set
column naming the tag; a video in more than one set (same file, same model)
appears once, its video_set listing every set.

Usage:
    python judging/combine.py                    # newest run
    python judging/combine.py --stamp 20260926   # the run stamped 20260926
    python judging/combine.py --tag superstar-ai # newest run of the superstar-ai video set
    python judging/combine.py --tag superstar-ai --tag wp2026-jan-mar --tag superstar-wp-janmar
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from gemini import RESULTS_DIR, prompt_path, run_stamp

SCORING_DIR = RESULTS_DIR / "result_scoring"
GROUPING_DIR = RESULTS_DIR / "result_grouping"
OUT_DIR = RESULTS_DIR / "result_combined"
STAMPED = re.compile(r"^(?P<kind>results|groups)_(?P<model>[^_]+)_(?P<stamp>\d{8}(?:_\d{6})?)(?:_(?P<tag>[^.]+))?\.csv$")
EXCEL_CELL_LIMIT = 32767
LONG_TEXT_WIDTH = 40  # widest a column gets; longer text is cut off on screen, not in the cell
ROW_HEIGHT = 15       # Excel's standard single-line row height, in points
HEADER_HEIGHT = 30    # two lines, so long column names wrap


def latest(folder: Path, kind: str, stamp: str | None, tag: str | None = None) -> dict[str, Path]:
    """Newest timestamped CSV per model in a folder, optionally only stamps starting with `stamp`.
    Only files carrying `tag` (or, without one, untagged files)."""
    found: dict[str, tuple[str, Path]] = {}
    for path in folder.glob(f"{kind}_*.csv"):
        m = STAMPED.match(path.name)
        if (m and m["stamp"].startswith(stamp or "") and m["tag"] == tag
                and (m["model"] not in found or m["stamp"] > found[m["model"]][0])):
            found[m["model"]] = (m["stamp"], path)
    return {model: path for model, (_, path) in sorted(found.items())}


def run_id(kind: str, stamp: str) -> str:
    """scoring_0927_2315 from ("scoring", "20260927_231507"); scoring_0926 for a date-only stamp."""
    date, _, clock = stamp.partition("_")
    return f"{kind}_{date[4:]}" + (f"_{clock[:4]}" if clock else "")


def prompt_file(df: pd.DataFrame, kind: str, stamp: str) -> pd.Series:
    """The prompt file behind each row: the CSV's own column if it has one (removed from df, since
    the workbook shows it up front), else by run date."""
    by_date = prompt_path("scoring" if kind == "scoring" else "grouping", stamp[2:8] + "z").name
    own = {"scoring": "prompt_file", "grouping": "grouping_prompt_file"}[kind]
    if own in df.columns:
        return df.pop(own).fillna(by_date).replace("", by_date)
    return pd.Series(by_date, index=df.index)


# The headline score is "creative_score" in the current CSVs and "overall" in the 26 Sep ones.
SCORE_COLUMNS = ("creative_score", "overall")
LEAD_COLUMNS = ["file", "video_set", "run_id", "source_file", "prompt_file", "llm_model", "overall_score"]


def stack(files: list[Path], kind: str) -> pd.DataFrame:
    """All files' rows in one frame, starting with LEAD_COLUMNS: video set, run and prompt details, then
    the model beside its overall score. Every other column stays in the CSV's own order."""
    frames = []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        m = STAMPED.match(path.name)
        df = df.rename(columns={c: "overall_score" for c in SCORE_COLUMNS if c in df.columns})
        if m["tag"]:
            df["video_set"] = m["tag"]
        df["run_id"] = run_id(kind, m["stamp"])
        df["source_file"] = path.name
        df["prompt_file"] = prompt_file(df, kind, m["stamp"])
        df["llm_model"] = m["model"]
        lead = [c for c in LEAD_COLUMNS if c in df.columns]
        frames.append(df[lead + [c for c in df.columns if c not in lead]])
    return merge_overlaps(pd.concat(frames, ignore_index=True))


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """A video judged once but listed in two tagged sets (same file and model) keeps its first row;
    video_set then names every set it belongs to."""
    if "video_set" not in df.columns:
        return df
    key = ["file", "llm_model"]
    sets = df.groupby(key)["video_set"].agg(lambda s: ", ".join(dict.fromkeys(s)))
    df = df.drop_duplicates(key, keep="first").copy()
    df["video_set"] = [sets[k] for k in zip(df["file"], df["llm_model"])]
    return df.reset_index(drop=True)


def fit_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Excel cells hold at most 32,767 characters; trim the rare longer transcript."""
    note = " ...[cut to fit Excel; full text in results.jsonl]"
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].map(lambda v: v[:EXCEL_CELL_LIMIT - len(note)] + note
                              if isinstance(v, str) and len(v) > EXCEL_CELL_LIMIT else v)
    return df


def style(ws, df: pd.DataFrame) -> None:
    """Standard one-line rows: long text stays in the cell unwrapped (click a cell to read it all in the
    formula bar) instead of stretching every row to fit transcripts."""
    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="305496")
    one_line = Alignment(wrap_text=False, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font, cell.alignment = font, one_line
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = ROW_HEIGHT
    for cell in ws[1]:
        cell.font, cell.fill = bold, header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = HEADER_HEIGHT
    for i, col in enumerate(df.columns, 1):
        values = df[col].dropna().astype(str)
        longest = max([len(col), *(len(v) for v in values.head(200))]) if len(values) else len(col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, longest + 2), LONG_TEXT_WIDTH)
    ws.freeze_panes = "B2"  # keep header row and file name in view
    ws.auto_filter.ref = ws.dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", help="use the run whose timestamp starts with this, e.g. 20260926")
    parser.add_argument("--tag", action="append",
                        help="use the files of this tagged video set, e.g. superstar-ai (repeatable)")
    args = parser.parse_args()

    tags = args.tag or [None]
    scoring = [p for t in tags for p in latest(SCORING_DIR, "results", args.stamp, t).values()]
    grouping = [p for t in tags for p in latest(GROUPING_DIR, "groups", args.stamp, t).values()]
    if not scoring and not grouping:
        sys.exit("No timestamped CSVs found in result_scoring/ or result_grouping/"
                 + (f" tagged {', '.join(args.tag)}" if args.tag else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (f"sahiyya_analysis{'_' + '_'.join(args.tag) if args.tag else ''}"
                     f"{'_run-' + args.stamp if args.stamp else ''}_{run_stamp()}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for tab, files in (("Scoring", scoring), ("Grouping", grouping)):
            if not files:
                continue
            df = fit_for_excel(stack(files, tab.lower()))
            df.to_excel(writer, sheet_name=tab, index=False)
            style(writer.sheets[tab], df)
            print(f"{tab}: {len(df)} rows from " + ", ".join(p.name for p in files))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
