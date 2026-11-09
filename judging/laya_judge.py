"""Score the videos with Laya (api.impossibl.com), a fast question-answering evaluator.

Laya does not read a rubric prompt or write reviews; it answers questions about a
piece of text. So it only does scoring: each text-judgeable metric of the scoring
rubric is asked as a 5-level score question, and Laya's expected level (0-4) is
mapped back to 0-100 with the rubric's band midpoints. Conditional metrics
(offer, USP, objection, enquiry) are gated by a yes/no question first and count as
not_applicable when the answer is no. The metrics that need eyes or ears are left
out, exactly as for the other text judges (text_judge.VIDEO_ONLY_METRICS).

Questions, levels and the Laya model live in the newest
prompts/*_prompt_laya.json. Laya reads the same written evidence as the other text
judges: the evidence Gemini recorded from the full video (opening seconds, shot
log, transcript, claims, CTAs, product mentions).

Results are appended to judging/results/results.jsonl under the model name (e.g.
laya-multilingual) and written to result_scoring/results_<model>_<stamp>.csv, so
combine.py stacks them with the other judges. Section scores, weights and the
creative score use the same arithmetic as gemini.py (score_summary).

.env: IMPOSSIBL_API_KEY

Usage:
    python judging/laya_judge.py                 # every video with evidence
    python judging/laya_judge.py --limit 10
    python judging/laya_judge.py --force         # re-score already scored videos
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import gemini
from text_judge import DEFAULT_SOURCE, VIDEO_ONLY_METRICS

API_URL = "https://api.impossibl.com/v1/systemone"
VIDEO_ONLY_NOTE = "Not scored by text-only judges: needs the video itself."


def load_questions() -> tuple[dict, str, str]:
    path = gemini.prompt_path("laya", ext="json")
    text = path.read_text(encoding="utf-8")
    return json.loads(text), path.name, gemini.prompt_fingerprint(text)


def build_questions(spec: dict) -> dict:
    questions = {}
    for section, metrics in spec["metrics"].items():
        for metric, instructions in metrics.items():
            questions[metric] = {"type": "score", "instructions": instructions, "criteria": spec["levels"]}
    for metric, instructions in spec["gates"].items():
        questions[f"gate_{metric}"] = {"type": "noul", "instructions": instructions}
    questions.update(spec["labels"])
    return questions


def evidence_text(ev: dict) -> str:
    """The written evidence as one readable block (Laya takes plain text)."""
    parts = [
        f"DURATION: {ev['input_audit']['duration']}",
        "FIRST 3 SECONDS:\n" + "\n".join(f"{o['interval']} visual: {o['visual']} | text: {o['text']} | "
                                         f"dialogue: {o['dialogue']}" for o in ev["opening"]),
        "SHOTS:\n" + "\n".join(f"{s['shot_id']} {s['start']}-{s['end']} {s['people_actions']}"
                               + (f" | on-screen text: {s['on_screen_text']}" if s["on_screen_text"] else "")
                               + (f" | product: {s['product_presence']}" if s["product_presence"] else "")
                               for s in ev["shots"]),
        "TRANSCRIPT:\n" + ev["transcript"],
        "CLAIMS:\n" + "\n".join(f"{c['claim_id']} {c['time']} {c['content']}" for c in ev["claims"]),
        "CALLS TO ACTION:\n" + "\n".join(f"{c['cta_id']} {c['time']} {c['words']} -> {c['destination']}"
                                         for c in ev["ctas"]),
    ]
    return "\n\n".join(parts)


def call_laya(model: str, state: str, questions: dict) -> dict:
    for attempt in range(1, 5):
        try:
            r = requests.post(API_URL, timeout=300, json={"model": model, "state": state, "questions": questions},
                              headers={"Authorization": f"Bearer {os.environ['IMPOSSIBL_API_KEY']}"})
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            if attempt == 4:
                raise
        time.sleep(10 * attempt)
    raise RuntimeError("Laya kept failing after retries")


def to_score(level: float, spec: dict) -> int:
    """Expected level (0..4) -> 0-100 by interpolating the rubric band midpoints."""
    points = spec["level_scores"]
    level = max(0.0, min(len(points) - 1.0, level))
    lo = int(level)
    hi = min(lo + 1, len(points) - 1)
    return round(points[lo] + (points[hi] - points[lo]) * (level - lo))


def judgement(answers: dict, spec: dict) -> dict:
    """Laya's answers in the section/metric shape score_summary expects."""
    j = {}
    for section, metrics in gemini.SECTIONS.items():
        j[section] = {}
        for metric in metrics:
            if metric in VIDEO_ONLY_METRICS.get(section, []):
                j[section][metric] = {"score": None, "applicability": "not_observable", "rationale": VIDEO_ONLY_NOTE}
                continue
            gate = answers.get(f"gate_{metric}")
            if gate is not None and gate["noul"] < 0.5:
                j[section][metric] = {"score": None, "applicability": "not_applicable",
                                      "rationale": f"gate probability {gate['noul']:.2f}"}
                continue
            a = answers[metric]
            j[section][metric] = {"score": to_score(a["score"], spec), "applicability": "applicable",
                                  "laya_level": a["score"], "confidence": a.get("confidence")}
    stage = answers["funnel_stage"]
    j["funnel"] = {"primary_stage": stage["choice"], "probabilities": stage.get("probabilities", {}),
                   "confidence": stage.get("confidence")}
    j["primary_objective"], j["objective_provisional"] = "qualified_enquiries", True
    j["medical_claim_probability"] = answers["medical_claim"]["noul"]
    return j


CSV_COLUMNS = [
    "file", "creative_score", *gemini.SECTIONS, *gemini.ALL_METRICS,
    "medical_claim_probability", "funnel_TOFU_p", "funnel_MOFU_p", "funnel_BOFU_p", "mean_confidence",
    "funnel_stage", "filename_stage", "weights_used", "provisional", "coverage", "readiness",
    "judged_at", "prompt_file", "prompt_version", "evidence_source", "laya_model",
]


def csv_rows(records: dict, model: str) -> list[dict]:
    rows = []
    for (file, m), rec in records.items():
        if m != model:
            continue
        j = rec["judgement"]
        probs = j["funnel"]["probabilities"]
        confs = [j[s][x]["confidence"] for s, ms in gemini.SECTIONS.items() for x in ms
                 if j[s][x].get("confidence") is not None]
        rows.append({
            "file": file, "creative_score": rec["creative_score"], **rec["sections"],
            **{x: j[s][x]["score"] for s, ms in gemini.SECTIONS.items() for x in ms},
            "medical_claim_probability": round(j["medical_claim_probability"], 3),
            **{f"funnel_{k}_p": round(probs.get(k, 0), 3) for k in ("TOFU", "MOFU", "BOFU")},
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else "",
            "funnel_stage": rec["funnel_stage"], "filename_stage": rec["filename_stage"],
            "weights_used": rec["weights_used"], "provisional": rec["provisional"],
            "coverage": " ".join(f"{k}={c}" for k, c in rec["coverage"].items()), "readiness": rec["readiness"],
            "judged_at": rec["judged_at"], "prompt_file": rec["prompt_file"], "prompt_version": rec["prompt_version"],
            "evidence_source": rec["evidence_source"], "laya_model": rec["laya_model"],
        })
    rows.sort(key=lambda r: (r["creative_score"] is not None, r["creative_score"] or 0), reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="video files (default: every video with evidence)")
    parser.add_argument("--limit", type=int, help="score at most N videos")
    parser.add_argument("--force", action="store_true", help="re-score videos already scored")
    parser.add_argument("--workers", type=int, default=4, help="parallel calls (default 4)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"whose evidence to read (default {DEFAULT_SOURCE})")
    args = parser.parse_args()

    gemini.load_dotenv()
    if not os.environ.get("IMPOSSIBL_API_KEY"):
        sys.exit("Set IMPOSSIBL_API_KEY in .env")
    spec, prompt_file, prompt_version = load_questions()
    questions = build_questions(spec)
    label = spec["model"].split("/")[-1]  # e.g. laya-multilingual

    records = gemini.load_records()
    sources = {f: r for (f, m), r in records.items() if m == args.source}
    files = [p.name for p in args.paths] if args.paths else sorted(sources)
    files = [f for f in files if f in sources]
    todo = [f for f in files if args.force or (f, label) not in records]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{label}: {len(files)} videos with evidence, {len(todo)} to score "
          f"({len(questions)} questions from {prompt_file} {prompt_version}; {args.workers} at a time)", flush=True)

    lock = threading.Lock()
    state = {"done": 0, "failures": 0}

    def process(file: str) -> None:
        start = time.perf_counter()
        try:
            resp = call_laya(spec["model"], evidence_text(sources[file]["judgement"]["evidence"]), questions)
            j = judgement(resp["answers"], spec)
            summary = gemini.score_summary(j)
            failing = summary.pop("commercial_readiness_failures")
            rec = {
                "rubric_version": gemini.RUBRIC_VERSION, "prompt_file": prompt_file, "prompt_version": prompt_version,
                "evidence_source": f"text evidence recorded by {args.source} (did not watch the video)",
                "judged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "file": file, "model": label,
                "laya_model": resp.get("model", ""), "funnel_stage": j["funnel"]["primary_stage"],
                "filename_stage": gemini.funnel_stage_from_name(file) or "",
                **{k: v for k, v in summary.items() if k not in ("missing_section", "unobserved_metric")},
                "provisional": True,  # text-only judge: visual and production sections are never scored
                "release_status": "",  # Laya scores only; it does not make release decisions
                "readiness": ("commercial readiness failed: " + ", ".join(failing)) if failing else "",
                "seconds": round(time.perf_counter() - start, 1),
                "judgement": {**j, "laya_raw": resp["answers"]},
            }
            line = (f"score {rec['creative_score'] if rec['creative_score'] is not None else 'n/a':>5}  "
                    f"{rec['funnel_stage']}  medical {j['medical_claim_probability']:.2f}  ({rec['seconds']:.0f}s)")
        except Exception as e:
            rec, line = None, f"ERROR {type(e).__name__}: {str(e)[:300]}"
        with lock:
            state["done"] += 1
            if rec:
                with open(gemini.JSONL_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records[(file, label)] = rec
            else:
                state["failures"] += 1
            print(f"[{state['done']}/{len(todo)}] {file}\n    {line}", flush=True)

    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        list(pool.map(process, todo))

    if todo:
        out = gemini.csv_path(label, gemini.run_stamp())
        gemini.SCORING_DIR.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows(records, label))
        print(f"{label}: wrote {out}")
    print(f"\nDone. {state['failures']} failed.")
    return 1 if state["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
