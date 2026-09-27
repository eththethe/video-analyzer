"""Judge ad videos with a panel of Gemini models on Vertex AI (0-100, six sections).

Setup (once):
    pip install google-genai google-cloud-storage pydantic
    gcloud auth application-default login

.env (repo root):
    GOOGLE_CLOUD_PROJECT=gl-rnd-dev
    GOOGLE_CLOUD_LOCATION=global        # Gemini 3.x models live on the global endpoint
    GCS_BUCKET=testsadawee              # videos are uploaded here once and reused
    PANEL_MODELS=gemini-3.8-flash,gemini-3.1-pro-preview

Prompts (edit these, not this file):
    prompts/scoring_prompt.md   the system prompt; {context} is replaced by
    prompts/context.md          the account context shared with group.py

Usage:
    python judging/gemini.py                     # judge everything in vid/
    python judging/gemini.py vid/some.mp4        # judge specific files
    python judging/gemini.py --limit 3           # quick test run
    python judging/gemini.py --force             # re-judge already scored videos
    python judging/gemini.py --workers 2         # videos judged in parallel (default 3)

Every model in PANEL_MODELS watches every video and does the full job on its
own: evidence breakdown, scene-by-scene description, then the judgement defined
in the scoring prompt. Results per (video, model) are appended to
judging/results/results.jsonl (resumable: a failed model is retried on the next
run without redoing the others). Each run writes its own CSV per model,
judging/results/results_<model>_<YYYYmmdd_HHMMSS>.csv, holding every judged video
for that model (not just this run's), rewritten after every video and sorted by
creative score. Earlier runs' CSVs are never overwritten.

Every record carries judged_at and prompt_version, a short fingerprint of the
exact scoring prompt (context included), so score changes can be traced to a
prompt change rather than a video change.

Scoring (computed here from the model's metric scores, as the prompt defines it)
    section_score   = mean of the section's applicable, observed metrics
    creative_score  = weighted mean of section scores, weights by the model's
                      funnel stage (or the link_clicks override); null if the
                      stage is unclear
    release_status  = the model's call, downgraded to revise_before_testing when
                      an applicable cta_strength / click_motivation /
                      purchase_desire is 40 or below (commercial readiness check)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Literal

from google import genai
from google.cloud import storage
from google.genai import errors, types
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "vid"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSONL_PATH = RESULTS_DIR / "results.jsonl"

RUBRIC_VERSION = 6  # bump when the schema/rubric changes; old records are ignored
DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.1-pro-preview"
GCS_PREFIX = "video-analyzer"
RETRY_CODES = (429, 500, 503, 504)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=value lines, no dependency on python-dotenv."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.split(" #")[0].strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


def prompt_fingerprint(text: str) -> str:
    """Short stable ID for a prompt's exact text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_prompt(name: str, **values: str) -> str:
    """prompts/<name>.md with {context} and any other {key} placeholders filled in."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    values.setdefault("context", (PROMPTS_DIR / "context.md").read_text(encoding="utf-8").strip())
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


# --- Output schema (mirrors prompts/scoring_prompt.md) ------------------------

Confidence = Literal["high", "medium", "low"]


class Metric(BaseModel):
    # No ge/le here: with 27 metrics the constraints push the schema past Vertex's complexity limit.
    score: int | None = Field(description="Integer 0-100; null when not_applicable or not_observable")
    applicability: Literal["applicable", "not_applicable", "not_observable"]
    evidence_time: str = Field(description="Inspected time range or script lines, e.g. '00:00-00:48'")
    evidence: str = Field(description="What is seen/heard; quote Malay verbatim")
    rationale: str
    confidence: Confidence

    @field_validator("score")
    @classmethod
    def clamp(cls, v: int | None) -> int | None:
        return None if v is None else max(0, min(100, v))


class Action(BaseModel):
    timestamp: str
    edit: str = Field(description="The exact change")
    intended_effect: str


class SectionReview(BaseModel):
    insight: str = Field(description="One key insight for this section")
    root_cause: str = Field(description="Why the problem exists, not a restatement of the score")
    actions: list[Action] = Field(description="Prioritized action plan")


# Step 1 - evidence breakdown

class InputAudit(BaseModel):
    input_mode: Literal["full_video", "sampled_frames_with_audio", "transcript_only", "script_only", "partial_asset"]
    duration: str
    aspect_ratio: str
    modalities: str = Field(description="Which of visuals / audio / on-screen text were accessible")
    limits: str = Field(description="Missing material, timestamp precision, limits on completeness")


class OpeningSecond(BaseModel):
    interval: str = Field(description="00:00-00:01, 00:01-00:02 or 00:02-00:03")
    visual: str
    text: str
    dialogue: str
    sound_action: str


class Shot(BaseModel):
    shot_id: str = Field(description="S01, S02, ...")
    start: str
    end: str
    framing: str = Field(description="Framing and camera movement")
    people_actions: str
    setting: str
    on_screen_text: str = Field(description="Exact readable text/graphics; empty if none")
    audio: str
    product_presence: str


class ClaimEvent(BaseModel):
    claim_id: str = Field(description="C01, C02, ...")
    time: str
    channel: Literal["spoken", "on_screen_text", "visual_implication", "mixed"]
    content: str = Field(description="Exact words, or a description of the visual implication")
    claim_type: str = Field(description="health / benefit / diagnostic / safety / numerical / superiority / "
                                        "authority / certification / price / scarcity / offer")


class CtaEvent(BaseModel):
    cta_id: str = Field(description="A01, A02, ...")
    time: str
    words: str = Field(description="Verbatim")
    action: str
    destination: str = Field(description="Visible destination, or 'not shown'")


class Evidence(BaseModel):
    input_audit: InputAudit
    people: list[str] = Field(description="Named only if supplied or identified in the asset, else Presenter A/B")
    settings_and_props: list[str]
    opening: list[OpeningSecond]
    shots: list[Shot]
    transcript: str = Field(description="Full original-language transcript with [mm:ss] and speaker labels")
    product_mentions: list[str] = Field(description="Every product name mention and visible product interval")
    claims: list[ClaimEvent]
    ctas: list[CtaEvent]
    audio_and_editing: str = Field(description="Music, SFX, live speech/voice-over, audio issues, transitions, "
                                               "subtitle treatment, graphics, editing rhythm")


# Step 3 - judgement

class Positioning(BaseModel):
    audience: Literal["self_50_plus", "self_40_49", "adult_child_buyer", "existing_customer",
                      "general_adult", "other", "unknown"]
    main_pain_point: Literal["kidney_health", "blood_sugar", "blood_pressure", "cholesterol", "heart_health",
                             "energy_daily_function", "general_health", "none", "other", "unknown"]
    main_objection: Literal["efficacy_doubt", "safety_ingredients", "authenticity_seller", "halal_confidence",
                            "price_value", "quantity_duration", "usage_routine", "purchase_support",
                            "none", "other", "unknown"]
    persuasion_angle: Literal["education", "ingredient_differentiation", "professional_explanation",
                              "lived_experience", "family_motivation", "seller_reassurance", "halal_reassurance",
                              "value_justification", "promotion", "routine_adherence", "other", "unknown"]
    presenter_type: Literal["doctor_claimed", "nutritionist_claimed", "customer_claimed", "celebrity", "brand_host",
                            "actor", "animated_character", "voiceover_only", "none", "other", "unknown"]
    presenter_name: str = Field(description="Name only if supplied or identified in the asset, else 'Presenter A'")
    format: Literal["talking_head", "interview", "testimonial", "explainer", "dramatized_story",
                    "animated_reporter", "walkthrough", "demonstration", "comparison", "offer_announcement",
                    "montage", "other", "unknown"]
    primary_hook_type: Literal["question", "curiosity_gap", "myth_correction", "warning", "direct_callout",
                               "story_open", "demonstration", "authority_lead", "objection_lead", "offer_lead",
                               "pattern_interrupt", "none", "other", "unknown"]
    dominant_emotion: Literal["concern", "regret", "hope", "reassurance", "family_care", "curiosity", "urgency",
                              "frustration", "aspiration", "neutral", "other", "unknown"]
    primary_cta_type: Literal["learn_more", "watch_continue", "whatsapp_enquiry", "comment_keyword", "purchase",
                              "save_share", "follow", "none", "other", "unknown"]
    core_concept: str = Field(description="Literal phrase of at most 12 words, not a creative title")
    label_evidence: str = Field(description="Short literal description and evidence behind the labels; "
                                            "describe any 'other' values here")


class Funnel(BaseModel):
    primary_stage: Literal["TOFU", "MOFU", "BOFU", "unclear"]
    confidence: Confidence
    evidence: str = Field(description="The dominant viewer decision this video serves")
    secondary_stage: Literal["TOFU", "MOFU", "BOFU", "none"]


class Hooks(BaseModel):
    hook_text: str = Field(description="Verbatim speech + on-screen text in the first 3 seconds")
    opening_promise: str = Field(description="The exact promise, whether/when it is fulfilled, any mismatch")
    bridge_3_10s: str = Field(description="Seconds 3-10: does interest develop or disappear?")
    hook_strength: Metric
    hook_relevance: Metric
    hook_payoff: Metric


class FormatTopic(BaseModel):
    takeaway: str = Field(description="The one-sentence takeaway")
    comprehension_risk: str = Field(description="Where comprehension may break")
    topic_relevance: Metric
    message_clarity: Metric
    format_fit: Metric
    audience_language_fit: Metric
    persuasion_logic: Metric


class Visual(BaseModel):
    visual_issues: str = Field(description="Spelling mistakes, distorted AI imagery, fake-looking documents, "
                                           "text/speech/footage mismatch; 'none observed' if none")
    subtitle_readability: Metric
    visual_engagement: Metric
    framing_lighting: Metric


class Production(BaseModel):
    audio_quality: Metric
    editing_pacing: Metric
    polish: Metric


class Speech(BaseModel):
    presenter_fit_and_arc: str = Field(description="Whether the presenter helps this message; the emotional arc")
    speech_clarity: Metric
    delivery_energy: Metric
    authenticity: Metric
    emotional_resonance: Metric


class ProductSales(BaseModel):
    cta_strength: Metric
    click_motivation: Metric
    purchase_desire: Metric
    proof_trust: Metric
    offer_urgency: Metric
    product_visibility: Metric
    usp_translation: Metric
    objection_resolution: Metric
    enquiry_quality: Metric
    cta_count: int = Field(ge=0, description="Distinct CTA occurrences, merging simultaneous spoken/text versions")
    product_first_appearance: str
    brand_first_mention: str


class ClaimAudit(BaseModel):
    claim_id: str
    time: str
    content: str = Field(description="Exact words or visual description")
    explicit_or_implied: Literal["explicit", "implied"]
    claim_type: str
    evidence_status: str = Field(description="e.g. supported by supplied evidence / unsupported / verification_pending")
    risk_domains: list[Literal["medical_accuracy", "substantiation", "Malaysian_advertising", "platform"]]
    severity: Literal["none", "low", "moderate", "high", "critical"]
    reason: str
    needed: str = Field(description="What evidence or revision is needed")


class ComplianceFlag(BaseModel):
    claim_id: str
    time: str
    quote: str
    reason: str


class Risk(BaseModel):
    claim_audit: list[ClaimAudit] = Field(description="One row for EVERY claim ID from the evidence breakdown")
    compliance_flags: list[ComplianceFlag] = Field(description="Only concerning instances; [] if none detected")
    brand_safety_tier: Literal["LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN"]
    verification_status: Literal["verification_pending", "verified"]


class Edit(BaseModel):
    priority: int = Field(ge=1)
    timestamp: str
    current_issue: str
    replacement: str = Field(description="Exact replacement line/visual/edit, Malay where it is dialogue or captions")
    why: str
    effort: Literal["low", "medium", "high"]


class HookAlternative(BaseModel):
    hook_malay: str
    strengthens: Literal["stopping_power", "click_motivation", "buying_motivation"]


class ABTest(BaseModel):
    hypothesis: str
    variable: str
    treatment_a: str
    treatment_b: str
    primary_metric: str
    guardrail: str = Field(description="Downstream quality guardrail")


class Verdict(BaseModel):
    one_sentence: str
    commercial_verdict: Literal["likely_to_earn_attention_only", "has_a_credible_click_proposition",
                                "has_a_credible_buying_proposition", "insufficient_evidence"]
    commercial_verdict_reason: str = Field(description="Justification; a hypothesis, not measured performance")
    strengths: list[str] = Field(description="Up to three, with timestamps")
    weaknesses: list[str] = Field(description="Three most consequential, with likely mechanisms")
    edits: list[Edit] = Field(description="Up to five prioritized edits")
    hook_alternatives: list[HookAlternative] = Field(description="Three distinct assertive Malay hooks")
    recommended_cta: str = Field(description="Concrete action plus viewer payoff, in Malay")
    product_transition: str = Field(description="Improved product transition, or 'not needed'")
    ab_test: ABTest


class SectionReviews(BaseModel):
    hooks: SectionReview
    format_topic: SectionReview
    visual: SectionReview
    production: SectionReview
    speech: SectionReview
    product_sales: SectionReview


# The review is split into three calls because Vertex rejects the whole schema in
# one request as too complex. The order follows the prompt: observe, score, assess.

class Observation(BaseModel):
    """Call 1: Steps 1-2 plus clearance, positioning, funnel and objective from Step 3A."""
    evidence: Evidence
    detailed_description: str = Field(description="Step 2: scene-by-scene description")
    summary: str = Field(description="Neutral 2-3 sentence summary")
    content_clearance: Literal["relevant", "unrelated", "insufficient_evidence"]
    positioning: Positioning
    funnel: Funnel
    primary_objective: Literal["purchases", "qualified_enquiries", "link_clicks", "awareness"]
    objective_provisional: bool


class Scores(BaseModel):
    """Call 2: Step 3C metrics for the six sections."""
    hooks: Hooks
    format_topic: FormatTopic
    visual: Visual
    production: Production
    speech: Speech
    product_sales: ProductSales


class Assessment(BaseModel):
    """Call 3: section insights/actions (3C), statuses, claim and risk review (3E), final verdict (3F)."""
    assessment_status: Literal["complete", "provisional", "insufficient_evidence"]
    release_status: Literal["eligible_for_testing", "revise_before_testing", "hold_for_verification",
                            "not_a_finished_ad", "cannot_assess"]
    section_reviews: SectionReviews
    risk: Risk
    verdict: Verdict


class Judgement(Observation, Scores, Assessment):
    """All three calls merged; this is what gets stored. Never sent to the API."""


# Section -> its 0-100 metrics, in the prompt's section order 1-6.
SECTIONS: dict[str, list[str]] = {
    "hooks": ["hook_strength", "hook_relevance", "hook_payoff"],
    "format_topic": ["topic_relevance", "message_clarity", "format_fit", "audience_language_fit", "persuasion_logic"],
    "visual": ["subtitle_readability", "visual_engagement", "framing_lighting"],
    "production": ["audio_quality", "editing_pacing", "polish"],
    "speech": ["speech_clarity", "delivery_energy", "authenticity", "emotional_resonance"],
    "product_sales": ["cta_strength", "click_motivation", "purchase_desire", "proof_trust", "offer_urgency",
                      "product_visibility", "usp_translation", "objection_resolution", "enquiry_quality"],
}
ALL_METRICS = [m for ms in SECTIONS.values() for m in ms]

# Proposed internal weights from the scoring prompt (section order 1-6).
WEIGHTS: dict[str, list[int]] = {
    "TOFU": [30, 25, 10, 5, 15, 15],
    "MOFU": [20, 20, 5, 5, 15, 35],
    "BOFU": [20, 15, 5, 5, 10, 45],
}
LINK_CLICKS_WEIGHTS = [30, 20, 5, 5, 10, 30]
READINESS_METRICS = ["cta_strength", "click_motivation", "purchase_desire"]
STRICTER_THAN_REVISE = {"hold_for_verification", "not_a_finished_ad", "cannot_assess"}


# --- Vertex AI + Cloud Storage ----------------------------------------------

def make_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )


def upload_video(path: Path) -> str:
    """Upload to GCS once (skipped if already there) and return the gs:// URI."""
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("Set GCS_BUCKET in .env")
    blob = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT")) \
        .bucket(bucket_name).blob(f"{GCS_PREFIX}/{path.name}")
    if not blob.exists():
        blob.upload_from_filename(str(path), content_type="video/mp4", timeout=900)
    return f"gs://{bucket_name}/{blob.name}"


def generate(client: genai.Client, model: str, contents: list, system_prompt: str, schema: type[BaseModel]) -> BaseModel:
    """One structured Gemini call with retries on transient errors."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
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
            time.sleep(15 * 2 ** attempt)  # 15s, 30s, 60s, 120s
    if response.parsed is None:
        raise RuntimeError(f"{schema.__name__}: could not parse model output: {(response.text or '')[:500]}")
    return response.parsed


def judge_video(client: genai.Client, model: str, uri: str, path: Path, system_prompt: str) -> tuple[Judgement, float]:
    """Observe, score, then assess: three calls, each seeing the video and the earlier output."""
    start = time.perf_counter()
    video = types.Part.from_uri(file_uri=uri, mime_type="video/mp4")
    intro = (f"Filename: {path.name}\nInput mode: full video with audio. No campaign brief, product playbook, "
             "approved-claims list, benchmark creatives or performance data are supplied.\n\n")

    obs = generate(client, model, [video, intro + (
        "This review is done in three requests. In THIS request do only: STEP 1 (evidence breakdown), "
        "STEP 2 (scene-by-scene description and summary), and from STEP 3A the content_clearance, the "
        "positioning labels, the primary funnel stage and the primary objective. Metric scores, section "
        "insights, the claim audit and the verdict come in the next requests."
    )], system_prompt, Observation)

    scores = generate(client, model, [video, intro + (
        "Your STEP 1-2 output and STEP 3A labels for this video, from the previous request:\n"
        f"{obs.model_dump_json()}\n\n"
        "Now do STEP 3C scoring only: every metric in the six sections (score or null, applicability, "
        "evidence time, evidence, rationale, confidence) plus each section's descriptive fields. Judge "
        "against the funnel stage and objective above, and refer to its claim IDs and CTA IDs."
    )], system_prompt, Scores)

    partial = {**obs.model_dump(), **scores.model_dump()}
    arithmetic = score_summary(partial)
    assessment = generate(client, model, [video, intro + (
        "Your evidence, labels and metric scores for this video, from the previous requests:\n"
        f"{json.dumps(partial, ensure_ascii=False)}\n\n"
        f"STEP 3D aggregation, computed from those scores: {json.dumps(arithmetic)}\n\n"
        "Now complete the review: for each of the six sections the key insight, root cause and prioritized "
        "action plan (3C); assessment_status and release_status; the claim, medical-accuracy and platform "
        "risk review with a claim_audit row for EVERY claim ID (3E); and the actionable final verdict (3F)."
    )], system_prompt, Assessment)

    judgement = Judgement(**obs.model_dump(), **scores.model_dump(), **assessment.model_dump())
    return judgement, time.perf_counter() - start


# --- Scoring & output -------------------------------------------------------

def funnel_stage_from_name(name: str) -> str | None:
    upper = name.upper()
    for stage in ("TOFU", "MOFU", "BOFU"):
        if upper.startswith(stage) or f"{stage} " in upper or f"-{stage}" in upper:
            return stage
    return None


def score_summary(j: dict) -> dict:
    """Section scores, coverage and creative score exactly as the prompt's AGGREGATION defines them."""
    sections, coverage = {}, {}
    for section, metrics in SECTIONS.items():
        ms = [j[section][m] for m in metrics]
        applicable = [m for m in ms if m["applicability"] != "not_applicable"]
        observed = [m["score"] for m in ms if m["applicability"] == "applicable" and m["score"] is not None]
        sections[section] = round(mean(observed), 1) if observed else None
        coverage[section] = f"{len(observed)}/{len(applicable)}"

    stage = j["funnel"]["primary_stage"]
    if j["primary_objective"] == "link_clicks" and not j["objective_provisional"]:
        weights, weights_used = LINK_CLICKS_WEIGHTS, "link_clicks override"
    elif stage in WEIGHTS:
        weights, weights_used = WEIGHTS[stage], stage
    else:
        weights, weights_used = None, "none (stage unclear)"

    creative = None
    if weights:
        pairs = [(v, w) for v, w in zip(sections.values(), weights) if v is not None]
        if pairs:
            creative = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 1)

    failing = [f"{m} {j['product_sales'][m]['score']}" for m in READINESS_METRICS
               if j["product_sales"][m]["applicability"] == "applicable"
               and j["product_sales"][m]["score"] is not None and j["product_sales"][m]["score"] <= 40]
    return {
        "sections": sections,
        "coverage": coverage,
        "weights_used": weights_used,
        "creative_score": creative,
        "missing_section": any(v is None for v in sections.values()),
        "unobserved_metric": any(j[sec][m]["applicability"] == "not_observable"
                                 for sec, ms in SECTIONS.items() for m in ms),
        "commercial_readiness_failures": failing,
    }


def aggregate(j: Judgement) -> dict:
    """score_summary plus the provisional flag and the commercial readiness check on release status."""
    summary = score_summary(j.model_dump())
    release = j.release_status
    failing = summary.pop("commercial_readiness_failures")
    if failing and release not in STRICTER_THAN_REVISE:
        release = "revise_before_testing"
    return {
        **{k: v for k, v in summary.items() if k not in ("missing_section", "unobserved_metric")},
        "provisional": summary["missing_section"] or summary["unobserved_metric"] or j.assessment_status != "complete",
        "release_status": release,
        "readiness": ("commercial readiness failed: " + ", ".join(failing)) if failing else "",
    }


def to_record(path: Path, model: str, j: Judgement, seconds: float, prompt_version: str) -> dict:
    return {
        "rubric_version": RUBRIC_VERSION,
        "prompt_version": prompt_version,
        "judged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": path.name,
        "model": model,
        "funnel_stage": j.funnel.primary_stage,
        "filename_stage": funnel_stage_from_name(path.name) or "",
        **aggregate(j),
        "seconds": round(seconds, 1),
        "judgement": j.model_dump(),
    }


def load_records() -> dict[tuple[str, str], dict]:
    records = {}
    if JSONL_PATH.exists():
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("rubric_version") == RUBRIC_VERSION:
                        records[(rec["file"], rec["model"])] = rec  # later lines win (e.g. after --force)
    return records


def short(model: str) -> str:
    return model.removeprefix("gemini-")


# file, then every numeric column, then text: short labels first, long free text last.
CSV_COLUMNS = [
    "file",
    "creative_score", *SECTIONS, *ALL_METRICS, "cta_count", "compliance_flags_count",
    "judged_at", "prompt_version",
    "funnel_stage", "filename_stage", "stage_confidence", "primary_objective", "weights_used",
    "provisional", "coverage", "release_status", "readiness", "brand_safety_tier", "verification_status",
    "commercial_verdict", "content_clearance",
    "audience", "main_pain_point", "main_objection", "persuasion_angle", "presenter_type", "presenter_name",
    "format", "primary_hook_type", "dominant_emotion", "primary_cta_type", "core_concept",
    "hook_text", "takeaway", "summary", "verdict", "commercial_verdict_reason", "strengths", "weaknesses",
    *(f"{s}_insight" for s in SECTIONS),
    "prioritized_edits", "hook_alternatives", "recommended_cta", "product_transition", "ab_test",
    "compliance_flags", "claim_audit", "detailed_description", "shot_log", "transcript",
]


def csv_path(model: str, stamp: str) -> Path:
    return RESULTS_DIR / f"results_{model}_{stamp}.csv"


def section_insight(review: dict) -> str:
    actions = " ; ".join(f"[{a['timestamp']}] {a['edit']} -> {a['intended_effect']}" for a in review["actions"])
    return f"INSIGHT: {review['insight']}\nROOT CAUSE: {review['root_cause']}\nACTIONS: {actions}"


def model_rows(records: dict[tuple[str, str], dict], model: str) -> list[dict]:
    """One CSV row per video judged by this model, using only that model's output."""
    rows = []
    for (file, m), rec in records.items():
        if m != model:
            continue
        j = rec["judgement"]
        ev, ps, risk, v, pos = j["evidence"], j["product_sales"], j["risk"], j["verdict"], j["positioning"]
        rows.append({
            "file": file,
            "creative_score": rec["creative_score"],
            **rec["sections"],
            **{metric: j[sec][metric]["score"] for sec, ms in SECTIONS.items() for metric in ms},
            "cta_count": ps["cta_count"],
            "compliance_flags_count": len(risk["compliance_flags"]),
            "judged_at": rec.get("judged_at", ""),
            "prompt_version": rec.get("prompt_version", ""),
            "funnel_stage": rec["funnel_stage"],
            "filename_stage": rec["filename_stage"],
            "stage_confidence": j["funnel"]["confidence"],
            "primary_objective": j["primary_objective"] + (" (provisional)" if j["objective_provisional"] else ""),
            "weights_used": rec["weights_used"],
            "provisional": rec["provisional"],
            "coverage": " ".join(f"{k}={c}" for k, c in rec["coverage"].items()),
            "release_status": rec["release_status"],
            "readiness": rec["readiness"],
            "brand_safety_tier": risk["brand_safety_tier"],
            "verification_status": risk["verification_status"],
            "commercial_verdict": v["commercial_verdict"],
            "content_clearance": j["content_clearance"],
            **{k: pos[k] for k in ("audience", "main_pain_point", "main_objection", "persuasion_angle",
                                   "presenter_type", "presenter_name", "format", "primary_hook_type",
                                   "dominant_emotion", "primary_cta_type", "core_concept")},
            "hook_text": j["hooks"]["hook_text"],
            "takeaway": j["format_topic"]["takeaway"],
            "summary": j["summary"],
            "verdict": v["one_sentence"],
            "commercial_verdict_reason": v["commercial_verdict_reason"],
            "strengths": "\n".join(v["strengths"]),
            "weaknesses": "\n".join(v["weaknesses"]),
            **{f"{s}_insight": section_insight(j["section_reviews"][s]) for s in SECTIONS},
            "prioritized_edits": "\n".join(
                f"{e['priority']}. [{e['timestamp']}] {e['current_issue']} => {e['replacement']} "
                f"(why: {e['why']}; effort: {e['effort']})" for e in v["edits"]),
            "hook_alternatives": "\n".join(f"{h['hook_malay']} [{h['strengthens']}]" for h in v["hook_alternatives"]),
            "recommended_cta": v["recommended_cta"],
            "product_transition": v["product_transition"],
            "ab_test": "\n".join(f"{k}: {val}" for k, val in v["ab_test"].items()),
            "compliance_flags": "\n".join(
                f"[{c['claim_id']} {c['time']}] \"{c['quote']}\" - {c['reason']}" for c in risk["compliance_flags"]),
            "claim_audit": "\n".join(
                f"[{c['claim_id']} {c['time']}] {c['content']} | {c['explicit_or_implied']} {c['claim_type']} | "
                f"{c['evidence_status']} | {c['severity']} {'/'.join(c['risk_domains'])} | {c['reason']} | "
                f"needs: {c['needed']}" for c in risk["claim_audit"]),
            "detailed_description": j["detailed_description"],
            "shot_log": "\n".join(
                f"{s['shot_id']} [{s['start']}-{s['end']}] {s['framing']}: {s['people_actions']}"
                + (f" | TEXT: {s['on_screen_text']}" if s["on_screen_text"] else "")
                + (f" | AUDIO: {s['audio']}" if s["audio"] else "")
                for s in ev["shots"]),
            "transcript": ev["transcript"],
        })
    rows.sort(key=lambda r: (r["creative_score"] is not None, r["creative_score"] or 0), reverse=True)
    return rows


def write_csv(rows: list[dict], path: Path) -> bool:
    """Rewrite the CSV. Returns False if the file is locked (e.g. open in Excel)."""
    try:
        f = open(path, "w", newline="", encoding="utf-8-sig")  # -sig so Excel reads Malay text fine
    except PermissionError:
        return False
    with f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge ad videos with a Gemini panel on Vertex AI (0-100).")
    parser.add_argument("paths", nargs="*", type=Path, help="video files (default: all of vid/)")
    parser.add_argument("--limit", type=int, help="judge at most N videos")
    parser.add_argument("--force", action="store_true", help="re-judge videos already in results")
    parser.add_argument("--workers", type=int, default=3, help="videos judged in parallel (default 3)")
    args = parser.parse_args()

    load_dotenv()
    models = [m.strip() for m in os.environ.get("PANEL_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
    system_prompt = load_prompt("scoring_prompt")
    prompt_version = prompt_fingerprint(system_prompt)
    stamp = run_stamp()

    videos = args.paths or sorted(VIDEO_DIR.glob("*.mp4"))
    records = load_records()
    todo = []
    for v in videos:
        missing = [m for m in models if args.force or (v.name, m) not in records]
        if missing:
            todo.append((v, missing))
    already = len(videos) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(videos)} videos, {already} already judged, {len(todo)} to go "
          f"(panel: {', '.join(models)}; {args.workers} at a time; prompt {prompt_version})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    lock = threading.Lock()
    state = {"done": 0, "failures": 0}

    def process(path: Path, missing: list[str]) -> None:
        lines = []
        try:
            uri = upload_video(path)
            with ThreadPoolExecutor(len(missing)) as pool:
                futures = {m: pool.submit(judge_video, client, m, uri, path, system_prompt) for m in missing}
            for m, fut in futures.items():
                try:
                    judgement, seconds = fut.result()
                except Exception as e:
                    with lock:
                        state["failures"] += 1
                    lines.append(f"    {short(m):22s} ERROR {type(e).__name__}: {str(e)[:300]}")
                    continue
                rec = to_record(path, m, judgement, seconds, prompt_version)
                with lock:
                    with open(JSONL_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    records[(path.name, m)] = rec
                score = rec["creative_score"]
                lines.append(f"    {short(m):22s} score {score if score is not None else 'n/a':>5}  "
                             f"{rec['funnel_stage']}  {rec['release_status']}  "
                             f"risk {judgement.risk.brand_safety_tier}  ({seconds:.0f}s)")
        except Exception as e:
            with lock:
                state["failures"] += len(missing)
            lines.append(f"    ERROR {type(e).__name__}: {str(e)[:300]}")

        with lock:
            state["done"] += 1
            locked = [m for m in models if not write_csv(model_rows(records, m), csv_path(m, stamp))]
            print("\n".join([f"[{state['done']}/{len(todo)}] {path.name}", *lines]), flush=True)
            if locked:
                print(f"  (CSV open in another program, skipped update; results are safe in {JSONL_PATH.name})")

    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        for path, missing in todo:
            pool.submit(process, path, missing)

    print(f"\nDone. {state['failures']} model runs failed. Results:")
    for m in models:
        out = csv_path(m, stamp)
        if not write_csv(model_rows(records, m), out):
            out = csv_path(m, run_stamp())
            write_csv(model_rows(records, m), out)
            print(f"  ({csv_path(m, stamp).name} is open in another program; wrote a new copy instead.)")
        print(f"  {out}")
    return 1 if state["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
