"""Turn judged videos into a Meta creative grouping and deployment plan.

Prompt: prompts/grouping_prompt.md ({context} comes from prompts/context.md,
{n} and {cards} are filled in here). Reads the judged videos from
judging/results/results.jsonl (run gemini.py first).

Two calls, because one call cannot reliably write 160+ inventory rows and a
full plan without dropping or repeating files:
    1. classify: STEP 1-2 of the prompt, i.e. the concept inventory (B) and the
       duplicate/variant decisions (C)
    2. plan: STEP 3-5, i.e. decision summary (A), launch plan (D), coverage
       matrix (E), content gaps (F), measurement plan (G) and the final file
       ledger (H), given the classification from call 1
Code then checks that every file appears exactly once in the inventory and the
ledger, that every LAUNCH file has one existing ad set, and prints any problem.

Output (each run gets its own timestamp, earlier runs are never overwritten):
    judging/results/groups_<model>_<YYYYmmdd_HHMMSS>.csv   one row per video: ledger status,
                                         ad set, launch role, concept and classification
    judging/results/groups_<model>_<YYYYmmdd_HHMMSS>.md    the plan to read: A, C, D, E, F, G
    judging/results/groups_<model>_<YYYYmmdd_HHMMSS>.json  everything the model returned
The .md and .json record the grouping prompt's fingerprint (prompt_version), and
the CSV records each video's scoring prompt_version.

Usage:
    python judging/group.py                          # plan with the last PANEL_MODELS model (Pro)
    python judging/group.py --model gemini-3.8-flash # plan with Flash, from Flash's judgements
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from typing import Literal

from google.genai import errors, types
from pydantic import BaseModel, Field

from gemini import (DEFAULT_MODELS, RESULTS_DIR, RETRY_CODES, load_dotenv, load_prompt, load_records, make_client,
                    prompt_fingerprint, run_stamp)

Stage = Literal["TOFU", "MOFU", "BOFU", "unclear"]
Role = Literal["A_relevance", "B_understanding", "C_differentiation", "D_credibility", "E_authenticity_service",
               "F_suitability_use", "G_value_quantity", "H_action", "none"]


# --- Call 1: classification (sections B and C) --------------------------------

class InventoryRow(BaseModel):
    file: str = Field(description="Exact input filename")
    concept_id: str
    topic: str
    takeaway: str = Field(description="One-sentence core takeaway")
    intended_buyer: str
    prior_knowledge: str = Field(description="Assumed prior product knowledge")
    primary_stage: Stage
    secondary_stage: str = Field(description="Only when justified, else 'none'")
    primary_role: Role
    secondary_roles: list[Role] = Field(description="At most two")
    objection: str = Field(description="The objection or belief addressed, in the customer's words")
    promise_proof_offer_cta: str = Field(description="Main promise, evidence/proof, offer, CTA and destination if known")
    format_presenter_emotion: str
    stands_alone: bool
    concerns: str = Field(description="Relevant scores, evidence limitations, suitability or claim concerns")


class Concept(BaseModel):
    concept_id: str
    name: str
    topic: str
    takeaway: str
    files: list[str]


class DuplicateDecision(BaseModel):
    files: list[str]
    relation: Literal["near_duplicate", "meaningful_variant"]
    preferred: str = Field(description="Preferred launch version")
    reserve: list[str]
    evidence: str
    confidence: Literal["high", "medium", "low"]
    verification: Literal["unverified_from_descriptions", "verified"]


class Classification(BaseModel):
    inventory: list[InventoryRow] = Field(description="One row per exact input filename")
    concepts: list[Concept]
    duplicates: list[DuplicateDecision]


# --- Call 2: plan (sections A, D, E, F, G, H) ---------------------------------

class AdSet(BaseModel):
    ad_set_id: str = Field(description="AS1, AS2, ...")
    campaign_purpose: str
    audience: str = Field(description="Proposed inclusions, lookback windows, exclusions; provisional where data is missing")
    objective_event: str
    budget_approach: str
    why_separate: str = Field(description="The operational reason this ad set exists")


class LaunchRow(BaseModel):
    ad_set_id: str
    file: str
    role: Role
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
    ad_set_id: str = Field(description="Exactly one primary ad set for LAUNCH, else empty")
    reason: str


class Plan(BaseModel):
    decision_summary: str = Field(description="A: recommended structure and rationale")
    assumptions: list[str] = Field(description="Key assumptions and missing information")
    ad_sets: list[AdSet]
    launch_plan: list[LaunchRow] = Field(description="D: one row per selected file")
    coverage: list[CoverageCell] = Field(description="E: ad sets by persuasion role")
    gaps: list[Gap] = Field(description="F: ranked content gaps")
    measurement_plan: str = Field(description="G: measurement and evidence-to-action plan")
    ledger: list[LedgerRow] = Field(description="H: every input file exactly once")


def card(rec: dict) -> str:
    j = rec["judgement"]
    ps = j["product_sales"]
    return json.dumps({
        "file": rec["file"],
        "funnel_stage": rec["funnel_stage"],
        "filename_funnel_label": rec["filename_stage"],
        "creative_score": rec["creative_score"],
        "section_scores": rec["sections"],
        "click_motivation": ps["click_motivation"]["score"],
        "purchase_desire": ps["purchase_desire"]["score"],
        "release_status": rec["release_status"],
        "commercial_readiness": rec["readiness"],
        "brand_safety_tier": j["risk"]["brand_safety_tier"],
        "compliance_flags": [f"{c['time']} \"{c['quote']}\": {c['reason']}" for c in j["risk"]["compliance_flags"]],
        "commercial_verdict": j["verdict"]["commercial_verdict"],
        "positioning": j["positioning"],
        "takeaway": j["format_topic"]["takeaway"],
        "hook_text": j["hooks"]["hook_text"],
        "ctas": [f"{c['time']} {c['words']}" for c in j["evidence"]["ctas"]],
        "duration": j["evidence"]["input_audit"]["duration"],
        "summary": j["summary"],
        "detailed_description": j["detailed_description"],
    }, ensure_ascii=False)


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
    if response.parsed is None:
        raise RuntimeError(f"{schema.__name__}: could not parse model output: {(response.text or '')[:500]}")
    return response.parsed


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
    ad_sets = {a.ad_set_id for a in p.ad_sets}
    for r in p.ledger:
        if r.status == "LAUNCH" and r.ad_set_id not in ad_sets:
            problems.append(f"{r.file} is LAUNCH but its ad set {r.ad_set_id!r} does not exist")
    launch = {r.file: r.ad_set_id for r in p.ledger if r.status == "LAUNCH"}
    for row in p.launch_plan:
        if row.file not in launch:
            problems.append(f"{row.file} is in the launch plan but not LAUNCH in the ledger")
        elif launch[row.file] != row.ad_set_id:
            problems.append(f"{row.file} is in {row.ad_set_id} in the launch plan but {launch[row.file]} in the ledger")
    return problems


def report(model: str, n: int, c: Classification, p: Plan, problems: list[str], meta: dict) -> str:
    """Markdown version of the plan for reading."""
    lines = [f"# Creative grouping plan ({model}, {n} videos)", "",
             f"Generated {meta['generated_at']} | grouping prompt {meta['prompt_version']} | "
             f"scoring prompts {', '.join(meta['scoring_prompt_versions']) or 'unknown'}", "",
             "## A. Decision summary", "", p.decision_summary,
             "", "**Assumptions**", "", *(f"- {a}" for a in p.assumptions), "", "## D. Launch plan", ""]
    for a in p.ad_sets:
        rows = [r for r in p.launch_plan if r.ad_set_id == a.ad_set_id]
        lines += [f"### {a.ad_set_id}: {a.campaign_purpose}", "",
                  f"- **Audience:** {a.audience}", f"- **Objective/event:** {a.objective_event}",
                  f"- **Budget:** {a.budget_approach}", f"- **Why separate:** {a.why_separate}", "",
                  "| File | Role | Why it earns a slot |", "|---|---|---|",
                  *(f"| {r.file} | {r.role} | {r.reason} |" for r in rows), ""]
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
    args = parser.parse_args()

    load_dotenv()
    panel = [m.strip() for m in os.environ.get("PANEL_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
    model = args.model or panel[-1]
    source = args.source or model
    recs = {file: rec for (file, m), rec in load_records().items() if m == source}
    if not recs:
        sys.exit(f"No judged videos from {source} in results.jsonl; run gemini.py first")

    ordered = sorted(recs.values(), key=lambda r: r["file"])
    stamp = run_stamp()
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "source": source,
        "prompt_version": prompt_fingerprint(load_prompt("grouping_prompt")),  # template + context, before cards
        "scoring_prompt_versions": sorted({r.get("prompt_version", "") for r in ordered} - {""}),
    }
    prompt = load_prompt("grouping_prompt", n=str(len(recs)), cards="\n\n".join(card(r) for r in ordered))
    client = make_client()

    print(f"Classifying {len(recs)} videos judged by {source}, using {model} ...", flush=True)
    c = generate(client, model, prompt + (
        "\n\nTHIS REQUEST: do STEP 1 and STEP 2 only. Return the classification/concept inventory (output B, "
        "one row per exact input filename), the concept families, and the duplicate/variant decisions "
        "(output C). The launch plan and ledger come in the next request."
    ), Classification)

    print("Writing the plan ...", flush=True)
    p = generate(client, model, prompt + (
        "\n\nYOUR STEP 1-2 OUTPUT FROM THE PREVIOUS REQUEST:\n" + c.model_dump_json() +
        "\n\nTHIS REQUEST: do STEP 3 to STEP 5 and return outputs A, D, E, F, G and H. The final ledger (H) "
        "must list every one of the input files exactly once."
    ), Plan)

    problems = check(set(recs), c, p)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {ext: RESULTS_DIR / f"groups_{model}_{stamp}.{ext}" for ext in ("csv", "md", "json")}
    out["json"].write_text(json.dumps({"meta": meta, "classification": c.model_dump(), "plan": p.model_dump(),
                                       "problems": problems}, ensure_ascii=False, indent=2), encoding="utf-8")
    out["md"].write_text(report(model, len(recs), c, p, problems, meta), encoding="utf-8")

    inv = {r.file: r for r in c.inventory}
    ledger = {r.file: r for r in p.ledger}
    launch = {r.file: r for r in p.launch_plan}
    concept_name = {x.concept_id: x.name for x in c.concepts}
    dup_note = {}
    for d in c.duplicates:
        for f in d.files:
            dup_note[f] = f"{d.relation}: preferred {d.preferred}" if f != d.preferred else f"{d.relation}: preferred version"
    rows = []
    for f in (r["file"] for r in ordered):
        led, i, lr, rec = ledger.get(f), inv.get(f), launch.get(f), recs[f]
        rows.append({
            "file": f,
            "creative_score": rec["creative_score"],
            "status": led.status if led else "MISSING FROM LEDGER",
            "ad_set_id": led.ad_set_id if led else "",
            "launch_role": lr.role if lr else "",
            "reason": led.reason if led else "",
            "why_in_ad_set": lr.reason if lr else "",
            "concept_id": i.concept_id if i else "",
            "concept": concept_name.get(i.concept_id, "") if i else "",
            "duplicate_note": dup_note.get(f, ""),
            "primary_stage": i.primary_stage if i else "",
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
            "scoring_prompt_version": rec.get("prompt_version", ""),
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
        print(f"  {a.ad_set_id}: {a.campaign_purpose} ({sum(r.ad_set_id == a.ad_set_id for r in p.launch_plan)} ads)")
    for x in problems:
        print(f"  WARNING: {x}")
    print(f"Wrote {out['csv']}, {out['md'].name} and {out['json'].name}")


if __name__ == "__main__":
    main()
