"""Turn judged videos into a Meta creative grouping and deployment plan.

Prompt: the newest prompts/*_prompt_grouping.md ({context} comes from the matching
*_prompt_context.md, {n} and {cards} are filled in here). Reads the judged videos from
judging/results/results.jsonl (run gemini.py first).

Split into several calls, because one answer cannot hold 160+ detailed
inventory rows plus a plan (it gets cut off at the output length limit):
    1. classify, in batches of BATCH_SIZE videos: STEP 1 of the prompt, one
       inventory row per video (B); saved to *_inventory.json straight away
       (--inventory reuses it)
    2. families, once per disease: STEP 2, concept families and the
       duplicate/variant decisions (C); concept IDs are merged into the rows
    3. plan: STEP 3-5, i.e. decision summary (A), launch plan (D), coverage
       matrix (E), content gaps (F) and measurement plan (G)
    4. ledger (H): LAUNCH rows from the launch plan, the rest in batches of
       LEDGER_BATCH files
Code then checks that every file appears exactly once in the inventory and the
ledger, that every LAUNCH file has one existing ad set, that there are at least
MIN_AD_SETS ad sets of AD_SET_SIZE ads with the STAGE_MIX (1 TOFU, 2 MOFU,
2 BOFU, from the playbook in prompt 261004a), and prints any problem.

Cards leave out release status, compliance flags and brand-safety tier (the
team does not use them for grouping). Classification sees the full card; the
whole-library calls get a compact card plus the classification.

Output (each run gets its own timestamp, earlier runs are never overwritten):
    judging/results/result_grouping/groups_<model>_<YYYYmmdd_HHMMSS>.csv   one row per video: ledger status,
                                         ad set, launch role, concept and classification
    judging/results/result_grouping/groups_<model>_<YYYYmmdd_HHMMSS>.md    the plan to read: A, C, D, E, F, G
    judging/results/result_grouping/groups_<model>_<YYYYmmdd_HHMMSS>.json  everything the model returned
The .md and .json record the grouping prompt file and fingerprint, and the CSV
records the grouping prompt file plus each video's scoring prompt file.

Usage:
    python judging/group.py                          # plan with the last PANEL_MODELS model (Pro)
    python judging/group.py --model gemini-3.8-flash # plan with Flash, from Flash's judgements
    python judging/group.py --gcs "SUPERSTAR AI/" --exclude "SUPERSTAR AI/Awareness/" --tag superstar-ai
                                                     # plan only the videos in those bucket folders (the
                                                     # same flags as gemini.py); files get a _<tag> suffix
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from google.genai import errors, types
from pydantic import BaseModel, Field

from gemini import (DEFAULT_MODELS, RESULTS_DIR, RETRY_CODES, gcs_videos, load_dotenv, load_prompt, load_records,
                    make_client, prompt_fingerprint, prompt_path, run_stamp)

GROUPING_DIR = RESULTS_DIR / "result_grouping"
BATCH_SIZE = 30  # videos per classification call; keeps each answer well under the output limit
CLASSIFY_WORKERS = 4  # classification calls in flight at once
AD_SET_SIZE = 5
MIN_AD_SETS = 5
STAGE_MIX = {"TOFU": 1, "MOFU": 2, "BOFU": 2}  # per ad set, allocated from the playbook (prompt 261004a)

Stage = Literal["TOFU", "MOFU", "BOFU", "unclear"]
Role = Literal["A_relevance", "B_understanding", "C_differentiation", "D_credibility", "E_authenticity_service",
               "F_suitability_use", "G_value_quantity", "H_action", "none"]
# Categories from the Sahiyya+ Hook & CTA Playbook (prompt 261004a onwards).
Disease = Literal["buah_pinggang", "3_serangkai", "kencing_manis", "kolesterol", "jantung_salur_darah",
                  "general_promo"]
HookPattern = Literal["silent_danger", "symptom_callout", "testimonial_real_person", "stage_or_number",
                      "question_faq", "objection_flipped", "usage_mistakes_repeat", "price_or_promo_first",
                      "visual_demo", "event_or_seasonal", "fear_of_burden", "family_skit", "other"]
ClosePattern = Literal["plain_instruction", "stock_or_price_warning", "deadline", "quantity_cap", "gift_or_bonus",
                       "soft_no_push", "none", "other"]
VideoType = Literal["animated_mascot", "presenter_warning_sign", "presenter_bundle_pitch", "presenter_retells_result",
                    "presenter_replies_comment", "ori_hq_check", "real_customer_testimonial", "product_broll_text",
                    "static_poster", "celebrity", "ai_realistic_person", "event_crowd", "other"]


# --- Call 1: classification (sections B and C) --------------------------------

class BatchRow(BaseModel):
    file: str = Field(description="Exact input filename")
    topic: str
    takeaway: str = Field(description="One-sentence core takeaway")
    intended_buyer: str
    prior_knowledge: str = Field(description="Assumed prior product knowledge")
    primary_stage: Stage
    secondary_stage: str = Field(description="Only when justified, else 'none'")
    disease: Disease = Field(description="Playbook disease focus; darah tinggi counts as 3_serangkai")
    hook_pattern: HookPattern
    close_pattern: ClosePattern
    close_names_person: bool = Field(description="The close names a person, e.g. 'Iqa akan bantu semak'")
    video_type: VideoType
    playbook_fit: str = Field(description="Formula parts and patterns it follows, 'stop doing' items it contains, "
                                          "whether its hook/close pairing is proven")
    primary_role: Role
    secondary_roles: list[Role] = Field(description="At most two")
    objection: str = Field(description="The objection or belief addressed, in the customer's words")
    promise_proof_offer_cta: str = Field(description="Main promise, evidence/proof, offer, CTA and destination if known")
    format_presenter_emotion: str
    stands_alone: bool
    concerns: str = Field(description="Relevant scores, evidence limitations, suitability or claim concerns")


class InventoryBatch(BaseModel):
    inventory: list[BatchRow] = Field(description="One row per exact input filename in this request")


class InventoryRow(BatchRow):
    concept_id: str


class Concept(BaseModel):
    concept_id: str
    name: str = Field(description="At most 8 words")
    topic: str = Field(description="At most 8 words")
    takeaway: str = Field(description="At most 20 words")
    files: list[str]


class DuplicateDecision(BaseModel):
    files: list[str]
    relation: Literal["near_duplicate", "meaningful_variant"]
    preferred: str = Field(description="Preferred launch version")
    reserve: list[str]
    evidence: str
    confidence: Literal["high", "medium", "low"]
    verification: Literal["unverified_from_descriptions", "verified"]


class Families(BaseModel):
    concepts: list[Concept] = Field(description="Every input file in exactly one concept")
    duplicates: list[DuplicateDecision]


class Classification(BaseModel):
    inventory: list[InventoryRow] = Field(description="One row per exact input filename")
    concepts: list[Concept]
    duplicates: list[DuplicateDecision]


# --- Call 2: plan (sections A, D, E, F, G, H) ---------------------------------

class AdSet(BaseModel):
    ad_set_id: str = Field(description="AS1, AS2, ...")
    disease_focus: str
    stage_mix: str = Field(description="e.g. '1 TOFU, 2 MOFU, 2 BOFU', with the reason for any swapped slot")
    campaign_purpose: str
    audience: str = Field(description="Proposed inclusions, lookback windows, exclusions; provisional where data is missing")
    objective_event: str
    budget_approach: str
    why_separate: str = Field(description="The operational reason this ad set exists")
    copy_guidance: str = Field(description="Caption opener type, CTA line and headline type from the playbook, "
                                           "each with a short Malay example")


class LaunchRow(BaseModel):
    ad_set_id: str
    file: str
    stage: Literal["TOFU", "MOFU", "BOFU"] = Field(description="The slot's stage; equals the video's primary stage")
    role: Role
    playbook_fit: str = Field(description="Hook/close pattern, video type and why they fit this disease and stage")
    reason: str = Field(description="Why it earns a slot and what it adds beyond the others")


class CoverageCell(BaseModel):
    ad_set_id: str
    role: Role
    files: list[str]
    strength: Literal["strong", "partial", "missing"]
    justification: str


class Gap(BaseModel):
    rank: int
    title: str = Field(description="NEW CONTENT brief title")
    audience: str
    belief_change: str
    required_proof: str
    cta: str
    priority_reason: str = Field(description="Likely purchase friction; say if it is a hypothesis")


class LedgerRow(BaseModel):
    file: str
    concept_id: str
    status: Literal["LAUNCH", "RESERVE", "REVISE", "HOLD"]
    ad_set_id: str = Field(description="For LAUNCH every ad set the file is in, comma-separated; else empty")
    reason: str


class LedgerBatch(BaseModel):
    ledger: list[LedgerRow] = Field(description="One row per exact input filename in this request; at most 25 "
                                                "words of reason each")


class PlanCore(BaseModel):
    """Call 3: everything but the ledger, which is written in batches (call 4) so 300+ rows fit."""
    decision_summary: str = Field(description="A: recommended structure and rationale")
    assumptions: list[str] = Field(description="Key assumptions and missing information")
    ad_sets: list[AdSet]
    launch_plan: list[LaunchRow] = Field(description="D: one row per selected file")
    coverage: list[CoverageCell] = Field(description="E: ad sets by persuasion role")
    gaps: list[Gap] = Field(description="F: ranked content gaps")
    measurement_plan: str = Field(description="G: measurement and evidence-to-action plan")


class Plan(PlanCore):
    """Calls 3 and 4 merged; what gets stored. Never sent to the API."""
    ledger: list[LedgerRow] = Field(description="H: every input file exactly once")


def card(rec: dict, compact: bool = False) -> str:
    """One video as the grouping model sees it. Release status, compliance flags and brand-safety tier are
    left out on purpose: from prompt 261004a the team does not use them for grouping. `compact` drops the long
    text for the whole-library calls, which also receive the STEP 1 inventory written from the full cards."""
    j = rec["judgement"]
    ps = j["product_sales"]
    c = {
        "file": rec["file"],
        "funnel_stage": rec["funnel_stage"],
        "filename_funnel_label": rec["filename_stage"],
        "creative_score": rec["creative_score"],
        "section_scores": rec["sections"],
        "click_motivation": ps["click_motivation"]["score"],
        "purchase_desire": ps["purchase_desire"]["score"],
        "hook_text": j["hooks"]["hook_text"],
        "ctas": [f"{c['time']} {c['words']}" for c in j["evidence"]["ctas"]],
        "duration": j["evidence"]["input_audit"]["duration"],
        "aspect_ratio": j["evidence"]["input_audit"]["aspect_ratio"],
    }
    if not compact:
        c |= {
            "commercial_verdict": j["verdict"]["commercial_verdict"],
            "positioning": j["positioning"],
            "takeaway": j["format_topic"]["takeaway"],
            "summary": j["summary"],
            "detailed_description": j["detailed_description"],
        }
    return json.dumps(c, ensure_ascii=False)


def generate(client, model: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                    max_output_tokens=65535,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            break
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == 4:
                raise
            time.sleep(15 * 2 ** attempt)
    finish = response.candidates[0].finish_reason if response.candidates else None
    if finish == types.FinishReason.MAX_TOKENS:
        raise RuntimeError(f"{schema.__name__}: the answer hit the output length limit and was cut off")
    if response.parsed is None:
        raise RuntimeError(f"{schema.__name__}: could not parse model output: {(response.text or '')[:500]}")
    return response.parsed


def classify(client, model: str, ordered: list[dict]) -> list[BatchRow]:
    """STEP 1 inventory rows, BATCH_SIZE videos per call, CLASSIFY_WORKERS calls at a time; files a batch
    skips get one more try."""
    rows: dict[str, BatchRow] = {}

    def run(batch: list[dict], label: str) -> list[BatchRow]:
        prompt = load_prompt("grouping", n=str(len(batch)), cards="\n\n".join(card(r) for r in batch))
        got = generate(client, model, prompt + (
            "\n\nTHIS REQUEST: do STEP 1 only, for these videos: one inventory row per exact input "
            "filename. Concept families, duplicates and the plan are decided later across the whole library."
        ), InventoryBatch)
        print(f"  classified {label}", flush=True)
        names = {r["file"] for r in batch}
        return [r for r in got.inventory if r.file in names]

    todo = list(ordered)
    for attempt in (1, 2):
        batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        print(f"Classifying {len(todo)} videos in {len(batches)} batches"
              + (" (retrying ones skipped earlier)" if attempt == 2 else "") + " ...", flush=True)
        with ThreadPoolExecutor(CLASSIFY_WORKERS) as pool:
            futures = [pool.submit(run, b, f"batch {n + 1}/{len(batches)}") for n, b in enumerate(batches)]
        for f in futures:
            rows.update({r.file: r for r in f.result()})
        todo = [r for r in ordered if r["file"] not in rows]
        if not todo:
            break
    return list(rows.values())


DISEASE_CODES = {"buah_pinggang": "BP", "3_serangkai": "3S", "kencing_manis": "KM", "kolesterol": "KO",
                 "jantung_salur_darah": "JT", "general_promo": "GP"}


def find_families(client, model: str, rows: list[BatchRow], recs: dict[str, dict]) -> Families:
    """STEP 2 once per disease, in parallel: one answer for 300+ files gets cut off at the output limit, and
    duplicates and variants share a disease anyway. Concept IDs get the disease code (BP-C1, GP-C1, ...) so
    they stay unique. A group that still overflows is split in two by file name."""

    def run(group: list[BatchRow], label: str) -> Families:
        files = sorted(r.file for r in group)
        prompt = load_prompt("grouping", n=str(len(files)), cards="\n\n".join(card(recs[f], compact=True) for f in files))
        try:
            fam = generate(client, model, prompt + (
                f"\n\nYOUR STEP 1 INVENTORY FOR THESE {len(files)} VIDEOS:\n"
                + json.dumps([r.model_dump() for r in group], ensure_ascii=False) +
                "\n\nTHIS REQUEST: do STEP 2 only, for these videos. Return the concept families (every one of "
                "these files in exactly one concept) and the duplicate/variant decisions (output C). Keep "
                "concept names, topics and takeaways short. The plan comes in a later request."
            ), Families)
        except RuntimeError as e:
            if "output length limit" not in str(e) or len(group) < 20:
                raise
            half = len(files) // 2
            print(f"  {label}: answer too long, splitting into two halves", flush=True)
            first = {f for f in files[:half]}
            parts = [run([r for r in group if r.file in first], label + " a"),
                     run([r for r in group if r.file not in first], label + " b")]
            for part, sub in zip(parts, ("a", "b")):
                for c in part.concepts:
                    c.concept_id = f"{sub}{c.concept_id}"
            return Families(concepts=[c for p in parts for c in p.concepts],
                            duplicates=[d for p in parts for d in p.duplicates])
        print(f"  families done: {label}", flush=True)
        return fam

    by_disease: dict[str, list[BatchRow]] = {}
    for r in rows:
        by_disease.setdefault(r.disease, []).append(r)
    with ThreadPoolExecutor(len(by_disease)) as pool:
        futures = {d: pool.submit(run, g, f"{d} ({len(g)} videos)") for d, g in sorted(by_disease.items())}
    concepts, duplicates = [], []
    for d, fut in futures.items():
        fam = fut.result()
        for c in fam.concepts:
            c.concept_id = f"{DISEASE_CODES.get(d, d)}-{c.concept_id}"
        concepts += fam.concepts
        duplicates += fam.duplicates
    return Families(concepts=concepts, duplicates=duplicates)


LEDGER_BATCH = 60


def launch_rows(plan: list[LaunchRow]) -> dict[str, list[LaunchRow]]:
    """Launch-plan rows per file. From prompt 261004b a video may fill a slot in more than one ad set."""
    by_file: dict[str, list[LaunchRow]] = {}
    for row in plan:
        by_file.setdefault(row.file, []).append(row)
    return by_file


def write_ledger(client, model: str, c: "Classification", core: PlanCore) -> list[LedgerRow]:
    """STEP H. LAUNCH rows come straight from the launch plan; every other file gets RESERVE, REVISE or HOLD
    from batched calls that see the plan and those files' inventory rows."""
    concept_of = {r.file: r.concept_id for r in c.inventory}
    ledger = {f: LedgerRow(file=f, concept_id=concept_of.get(f, ""), status="LAUNCH",
                           ad_set_id=", ".join(r.ad_set_id for r in rows),
                           reason=" | ".join(f"{r.ad_set_id}: {r.reason}" for r in rows) if len(rows) > 1 else rows[0].reason)
              for f, rows in launch_rows(core.launch_plan).items()}
    plan_view = json.dumps({
        "ad_sets": [{"ad_set_id": a.ad_set_id, "disease_focus": a.disease_focus, "stage_mix": a.stage_mix}
                    for a in core.ad_sets],
        "launch": [{"file": r.file, "ad_set_id": r.ad_set_id, "stage": r.stage} for r in core.launch_plan],
    }, ensure_ascii=False)
    base = load_prompt("grouping", n="0", cards="(not repeated here; see the inventory rows below)")

    def run(batch: list[InventoryRow]) -> list[LedgerRow]:
        names = {r.file for r in batch}
        dups = [d.model_dump() for d in c.duplicates if names & set(d.files)]
        got = generate(client, model, base + (
            f"\n\nTHE LAUNCH PLAN ALREADY DECIDED:\n{plan_view}\n\nDUPLICATE/VARIANT DECISIONS FOR THESE FILES:\n"
            f"{json.dumps(dups, ensure_ascii=False)}\n\nINVENTORY ROWS FOR THESE {len(batch)} FILES:\n"
            + json.dumps([r.model_dump() for r in batch], ensure_ascii=False) +
            "\n\nTHIS REQUEST: write final ledger (H) rows for exactly these files, none of which was selected "
            "for launch. Status RESERVE, REVISE or HOLD as defined in OUTPUT H, never LAUNCH; ad_set_id empty; "
            "concept_id as in the inventory; a reason of at most 25 words."
        ), LedgerBatch)
        return [r for r in got.ledger if r.file in names]

    for attempt in (1, 2):
        todo = [r for r in c.inventory if r.file not in ledger]
        if not todo:
            break
        batches = [todo[i:i + LEDGER_BATCH] for i in range(0, len(todo), LEDGER_BATCH)]
        print(f"Writing the ledger for {len(todo)} non-launch files in {len(batches)} batches"
              + (" (retrying ones skipped earlier)" if attempt == 2 else "") + " ...", flush=True)
        with ThreadPoolExecutor(CLASSIFY_WORKERS) as pool:
            futures = [pool.submit(run, b) for b in batches]
        for f in futures:
            for r in f.result():
                ledger.setdefault(r.file, r)
    return list(ledger.values())


def with_concepts(rows: list[BatchRow], concepts: list[Concept]) -> list[InventoryRow]:
    concept_of = {}
    for c in concepts:
        for f in c.files:
            concept_of.setdefault(f, c.concept_id)
    return [InventoryRow(**r.model_dump(), concept_id=concept_of.get(r.file, "")) for r in rows]


def check(files: set[str], c: Classification, p: Plan) -> list[str]:
    """Rule violations the model may have made; printed, not fatal."""
    problems = []
    for name, listed in (("inventory", [r.file for r in c.inventory]), ("ledger", [r.file for r in p.ledger])):
        for f, n in Counter(listed).items():
            if n > 1:
                problems.append(f"{f} appears {n} times in the {name}")
        for f in sorted(files - set(listed)):
            problems.append(f"{f} is missing from the {name}")
        for f in sorted(set(listed) - files):
            problems.append(f"{name} names an unknown file: {f}")
    for f in sorted(files - {f for x in c.concepts for f in x.files}):
        problems.append(f"{f} is in no concept family")
    ad_sets = {a.ad_set_id for a in p.ad_sets}
    # A LAUNCH ledger row lists its ad sets comma-separated (one video may be in several, prompt 261004b).
    launch = {r.file: {s.strip() for s in r.ad_set_id.split(",") if s.strip()} for r in p.ledger if r.status == "LAUNCH"}
    for f, sets in launch.items():
        if not sets:
            problems.append(f"{f} is LAUNCH but has no ad set")
        for s in sorted(sets - ad_sets):
            problems.append(f"{f} is LAUNCH but its ad set {s!r} does not exist")
    for f, rows in launch_rows(p.launch_plan).items():
        planned = {row.ad_set_id for row in rows}
        if f not in launch:
            problems.append(f"{f} is in the launch plan but not LAUNCH in the ledger")
        elif launch[f] != planned:
            problems.append(f"{f} is in {sorted(planned)} in the launch plan but {sorted(launch[f])} in the ledger")
    for f in sorted(set(launch) - {row.file for row in p.launch_plan}):
        problems.append(f"{f} is LAUNCH in the ledger but has no row in the launch plan")

    # Ad set shape from prompt 261004a: exactly AD_SET_SIZE ads, STAGE_MIX (or a neighbouring swap), never one stage.
    if len(p.ad_sets) < MIN_AD_SETS:
        problems.append(f"only {len(p.ad_sets)} ad sets; the plan asks for at least {MIN_AD_SETS}")
    stage_of = {r.file: r.primary_stage for r in c.inventory}
    for a in p.ad_sets:
        rows = [row for row in p.launch_plan if row.ad_set_id == a.ad_set_id]
        stages = Counter(row.stage for row in rows)
        for f, n in Counter(row.file for row in rows).items():
            if n > 1:
                problems.append(f"{f} is in {a.ad_set_id} {n} times")
        if len(rows) != AD_SET_SIZE:
            problems.append(f"{a.ad_set_id} has {len(rows)} ads, not {AD_SET_SIZE}")
        if len(stages) < 2:
            problems.append(f"{a.ad_set_id} uses a single stage: {dict(stages)}")
        elif dict(stages) != STAGE_MIX:
            problems.append(f"{a.ad_set_id} stage mix is {dict(stages)}, not {STAGE_MIX} (ok only if explained: "
                            f"{a.stage_mix})")
        for row in rows:
            if stage_of.get(row.file) not in (None, row.stage):
                problems.append(f"{row.file} fills a {row.stage} slot in {a.ad_set_id} but was classified "
                                f"{stage_of[row.file]}")
    return problems


def report(model: str, n: int, c: Classification, p: Plan, problems: list[str], meta: dict) -> str:
    """Markdown version of the plan for reading."""
    lines = [f"# Creative grouping plan ({model}, {n} videos)", "",
             f"Generated {meta['generated_at']} | grouping prompt {meta['prompt_file']} ({meta['prompt_version']}) | "
             f"scoring prompts {', '.join(meta['scoring_prompt_files']) or 'unknown'}", "",
             "## A. Decision summary", "", p.decision_summary,
             "", "**Assumptions**", "", *(f"- {a}" for a in p.assumptions), "", "## D. Launch plan", ""]
    stage_order = {"TOFU": 0, "MOFU": 1, "BOFU": 2}
    for a in p.ad_sets:
        rows = sorted((r for r in p.launch_plan if r.ad_set_id == a.ad_set_id), key=lambda r: stage_order[r.stage])
        lines += [f"### {a.ad_set_id}: {a.disease_focus} ({a.stage_mix})", "", f"**Purpose:** {a.campaign_purpose}", "",
                  f"- **Audience:** {a.audience}", f"- **Objective/event:** {a.objective_event}",
                  f"- **Budget:** {a.budget_approach}", f"- **Why separate:** {a.why_separate}",
                  f"- **Copy (playbook):** {a.copy_guidance}", "",
                  "| Stage | File | Role | Playbook fit | Why it earns a slot |", "|---|---|---|---|---|",
                  *(f"| {r.stage} | {r.file} | {r.role} | {r.playbook_fit} | {r.reason} |" for r in rows), ""]
    lines += ["## E. Coverage matrix", "", "| Ad set | Role | Strength | Files | Justification |", "|---|---|---|---|---|",
              *(f"| {x.ad_set_id} | {x.role} | {x.strength} | {', '.join(x.files)} | {x.justification} |"
                for x in p.coverage), "", "## C. Duplicates and variants", ""]
    for d in c.duplicates:
        lines.append(f"- **{d.relation}** ({d.confidence}, {d.verification}): preferred **{d.preferred}**; "
                     f"reserve {', '.join(d.reserve) or 'none'}. {d.evidence}")
    lines += ["", "## F. Content gaps (NEW CONTENT)", ""]
    for g in sorted(p.gaps, key=lambda g: g.rank):
        lines += [f"{g.rank}. **{g.title}** ({g.audience}): {g.belief_change} Proof needed: {g.required_proof}. "
                  f"CTA: {g.cta}. Why: {g.priority_reason}"]
    lines += ["", "## G. Measurement plan", "", p.measurement_plan, ""]
    status = Counter(r.status for r in p.ledger)
    lines += ["## H. Ledger", "", ", ".join(f"{k}: {v}" for k, v in status.most_common()),
              "(one row per file in the CSV)", ""]
    if problems:
        lines += ["## Checks that failed", "", *(f"- {x}" for x in problems), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model that writes the plan (default: last PANEL_MODELS entry)")
    parser.add_argument("--source", help="whose judgements to use (default: same as --model)")
    parser.add_argument("--gcs", action="append", metavar="PREFIX", help="only videos under this bucket folder (repeatable)")
    parser.add_argument("--exclude", action="append", metavar="PREFIX", help="skip videos under this bucket folder")
    parser.add_argument("--tag", help="name for this set of videos; required with --gcs, added to the file names")
    parser.add_argument("--inventory", type=Path, metavar="JSON",
                        help="reuse a saved STEP 1 inventory (*_inventory.json) instead of classifying again")
    args = parser.parse_args()
    if args.gcs and not args.tag:
        parser.error("--gcs needs --tag, so its files are told apart from the full-library plan")

    load_dotenv()
    panel = [m.strip() for m in os.environ.get("PANEL_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
    model = args.model or panel[-1]
    source = args.source or model
    recs = {file: rec for (file, m), rec in load_records().items() if m == source}
    if args.gcs:
        wanted = {v.name for v in gcs_videos(args.gcs, args.exclude)}
        if unjudged := sorted(wanted - set(recs)):
            sys.exit(f"{len(unjudged)} of {len(wanted)} videos not judged by {source} yet, e.g. {unjudged[0]}; "
                     "run gemini.py with the same --gcs first")
        recs = {f: r for f, r in recs.items() if f in wanted}
    if not recs:
        sys.exit(f"No judged videos from {source} in results.jsonl; run gemini.py first")

    ordered = sorted(recs.values(), key=lambda r: r["file"])
    stamp = run_stamp()
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "source": source,
        "prompt_file": prompt_path("grouping").name,
        "prompt_version": prompt_fingerprint(load_prompt("grouping")),  # template + context, before cards
        "scoring_prompt_files": sorted({r.get("prompt_file", "") for r in ordered} - {""}),
        "scoring_prompt_versions": sorted({r.get("prompt_version", "") for r in ordered} - {""}),
    }
    # The plan sees every video, so it gets compact cards; the STEP 1 inventory carries the rest.
    prompt = load_prompt("grouping", n=str(len(recs)), cards="\n\n".join(card(r, compact=True) for r in ordered))
    client = make_client()
    GROUPING_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = {ext: GROUPING_DIR / f"groups_{model}_{stamp}{suffix}.{ext}" for ext in ("csv", "md", "json")}

    rows = []
    if args.inventory:
        rows = [r for r in (BatchRow(**x) for x in json.loads(args.inventory.read_text(encoding="utf-8")))
                if r.file in recs]
        print(f"Reusing {len(rows)} classified videos from {args.inventory.name}", flush=True)
    have = {r.file for r in rows}
    if missing := [r for r in ordered if r["file"] not in have]:
        rows += classify(client, model, missing)
    # Saved straight away, so a later failure does not cost the classification (rerun with --inventory).
    inventory_path = GROUPING_DIR / f"groups_{model}_{stamp}{suffix}_inventory.json"
    inventory_path.write_text(json.dumps([r.model_dump() for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved the classification to {inventory_path.name}", flush=True)

    print("Finding concept families and duplicates, per disease ...", flush=True)
    fam = find_families(client, model, rows, recs)
    c = Classification(inventory=with_concepts(rows, fam.concepts), concepts=fam.concepts, duplicates=fam.duplicates)

    print("Writing the plan ...", flush=True)
    core = generate(client, model, prompt + (
        "\n\nYOUR STEP 1-2 OUTPUT FROM THE PREVIOUS REQUESTS:\n" + c.model_dump_json() +
        "\n\nTHIS REQUEST: do STEP 3 to STEP 5 and return outputs A, D, E, F and G. The final ledger (H) for "
        "the files you do not launch is written in later requests."
    ), PlanCore)
    p = Plan(**core.model_dump(), ledger=write_ledger(client, model, c, core))

    problems = check(set(recs), c, p)
    out["json"].write_text(json.dumps({"meta": meta, "classification": c.model_dump(), "plan": p.model_dump(),
                                       "problems": problems}, ensure_ascii=False, indent=2), encoding="utf-8")
    out["md"].write_text(report(model, len(recs), c, p, problems, meta), encoding="utf-8")

    inv = {r.file: r for r in c.inventory}
    ledger = {r.file: r for r in p.ledger}
    launch = launch_rows(p.launch_plan)
    concept_name = {x.concept_id: x.name for x in c.concepts}
    dup_note = {}
    for d in c.duplicates:
        for f in d.files:
            dup_note[f] = f"{d.relation}: preferred {d.preferred}" if f != d.preferred else f"{d.relation}: preferred version"
    def per_slot(slots: list[LaunchRow], attr: str) -> str:
        """One value per ad set the video is in: plain for one, "AS1: x | AS3: y" for several, "" for none."""
        if len(slots) == 1:
            return getattr(slots[0], attr)
        return " | ".join(f"{s.ad_set_id}: {getattr(s, attr)}" for s in slots)

    rows = []
    for f in (r["file"] for r in ordered):
        led, i, slots, rec = ledger.get(f), inv.get(f), launch.get(f, []), recs[f]
        rows.append({
            "file": f,
            "creative_score": rec["creative_score"],
            "status": led.status if led else "MISSING FROM LEDGER",
            "ad_set_id": led.ad_set_id if led else "",
            "slot_stage": per_slot(slots, "stage"),
            "launch_role": per_slot(slots, "role"),
            "reason": led.reason if led else "",
            "why_in_ad_set": per_slot(slots, "reason"),
            "slot_playbook_fit": per_slot(slots, "playbook_fit"),
            "concept_id": i.concept_id if i else "",
            "concept": concept_name.get(i.concept_id, "") if i else "",
            "duplicate_note": dup_note.get(f, ""),
            "primary_stage": i.primary_stage if i else "",
            "disease": i.disease if i else "",
            "hook_pattern": i.hook_pattern if i else "",
            "close_pattern": i.close_pattern if i else "",
            "close_names_person": i.close_names_person if i else "",
            "video_type": i.video_type if i else "",
            "playbook_fit": i.playbook_fit if i else "",
            "primary_role": i.primary_role if i else "",
            "secondary_roles": ", ".join(i.secondary_roles) if i else "",
            "intended_buyer": i.intended_buyer if i else "",
            "topic": i.topic if i else "",
            "takeaway": i.takeaway if i else "",
            "objection": i.objection if i else "",
            "stands_alone": i.stands_alone if i else "",
            "concerns": i.concerns if i else "",
            "release_status": rec["release_status"],
            "brand_safety_tier": rec["judgement"]["risk"]["brand_safety_tier"],
            "grouping_prompt_file": meta["prompt_file"],
            "scoring_prompt_file": rec.get("prompt_file", ""),
            "scoring_prompt_version": rec.get("prompt_version", ""),
            "short_summary": rec["judgement"].get("short_summary", ""),
        })
    order = {"LAUNCH": 0, "RESERVE": 1, "REVISE": 2, "HOLD": 3}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["ad_set_id"], -(r["creative_score"] or 0)))
    with out["csv"].open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    status = Counter(r["status"] for r in rows)
    print(f"{len(c.concepts)} concepts, {len(c.duplicates)} duplicate/variant groups, {len(p.ad_sets)} ad sets; "
          + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    for a in p.ad_sets:
        stages = Counter(r.stage for r in p.launch_plan if r.ad_set_id == a.ad_set_id)
        print(f"  {a.ad_set_id}: {a.disease_focus} ({sum(stages.values())} ads: "
              + ", ".join(f"{stages[s]} {s}" for s in ("TOFU", "MOFU", "BOFU")) + ")")
    for x in problems:
        print(f"  WARNING: {x}")
    print(f"Wrote {out['csv']}, {out['md'].name} and {out['json'].name}")


if __name__ == "__main__":
    main()
