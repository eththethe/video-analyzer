"""Export lifetime Meta ad performance for every ad account the token can see.

Writes two CSVs to judging/results/result_meta-ads/, stamped with the run time so earlier
exports are never overwritten:
    meta_ads_<YYYYmmdd_HHMMSS>.csv        one row per ad that delivered: account,
                        campaign, ad set, ad, video, delivery period,
                        spend/CTR/CPM/video/funnel metrics
    meta_ads_daily_<YYYYmmdd_HHMMSS>.csv  one row per ad per day (spend,
                        impressions, clicks), the source of first_day / last_day /
                        days_advertised

.env (repo root):
    META_ACCESS_TOKEN=...   # needs ads_read; see "Video links" below for pages

Usage:
    python judging/meta_ads.py                          # every ad account, lifetime
    python judging/meta_ads.py --accounts 420687623753634   # just these accounts
    python judging/meta_ads.py --refresh                # ignore cached API responses

API responses are cached per account in judging/results/meta_raw/, so a rerun
after a crash or rate limit only fetches what is missing. Use --refresh for
fresh numbers and fresh download links (Meta's links expire after a few days).

Video links
    download_link comes from the ad account's video library, or from the page
    that posted the video if the token can get that page's access token (the
    system user must be assigned to the page, and the token needs
    pages_read_engagement + pages_show_list). Videos that can't be read still get
    video_id and watch_link (opens on facebook.com for anyone with page access).
    Dynamic-creative ads can use several videos; their ids are joined with "|".
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from gemini import RESULTS_DIR, load_dotenv, run_stamp

API = "https://graph.facebook.com/v26.0"
RAW_DIR = RESULTS_DIR / "meta_raw"
META_DIR = RESULTS_DIR / "result_meta-ads"

ALL_STATUSES = ["ACTIVE", "PAUSED", "DELETED", "PENDING_REVIEW", "DISAPPROVED", "PREAPPROVED",
                "PENDING_BILLING_INFO", "CAMPAIGN_PAUSED", "ARCHIVED", "ADSET_PAUSED", "IN_PROCESS",
                "WITH_ISSUES"]
RATE_LIMIT_CODES = {4, 17, 32, 613, 80000, 80001, 80002, 80003, 80004, 80005, 80006, 80008, 80009, 80014}

INSIGHT_FIELDS = [
    "account_id", "account_name", "campaign_id", "campaign_name", "objective", "adset_id", "adset_name",
    "ad_id", "ad_name", "spend", "impressions", "reach", "frequency", "clicks", "ctr",
    "inline_link_clicks", "inline_link_click_ctr", "cpc", "cpm", "actions", "action_values",
    "purchase_roas", "video_thruplay_watched_actions", "video_p25_watched_actions",
    "video_p50_watched_actions", "video_p75_watched_actions", "video_p100_watched_actions",
    "video_avg_time_watched_actions",
]
DAILY_FIELDS = ["ad_id", "spend", "impressions", "clicks", "inline_link_clicks"]
AD_FIELDS = ("id,name,created_time,effective_status,creative{id,name,video_id,thumbnail_url,"
             "object_story_spec{page_id,video_data{video_id}},asset_feed_spec{videos{video_id}}}")

COLUMNS = [
    "account_id", "account_name", "currency", "campaign_id", "campaign_name", "objective",
    "adset_id", "adset_name", "ad_id", "ad_name", "ad_status", "ad_created",
    "first_day", "last_day", "days_advertised",
    "video_id", "video_file_name", "video_length_s", "download_link", "watch_link", "thumbnail_url",
    "spend", "impressions", "reach", "frequency", "cpm", "clicks", "ctr", "link_clicks", "link_ctr", "cpc",
    "video_3s_views", "hook_rate", "thruplays", "hold_rate", "p25_views", "p50_views", "p75_views",
    "p100_views", "avg_watch_s",
    "landing_page_views", "add_to_cart", "initiate_checkout", "purchases", "purchase_value", "roas",
    "cost_per_purchase", "leads", "cost_per_lead", "messaging_started",
]

session = requests.Session()


class TooMuchData(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def call(method: str, path: str, params: dict | None = None) -> dict:
    """One Graph API call with backoff on rate limits and transient errors."""
    params = dict(params or {}, access_token=os.environ["META_ACCESS_TOKEN"])
    for attempt in range(8):
        try:
            resp = session.request(method, f"{API}/{path}", params=params, timeout=120)
        except requests.RequestException as exc:
            wait = 2 ** attempt * 5
            log(f"  network error ({exc.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue
        body = resp.json()
        err = body.get("error")
        if not err:
            return body
        code = err.get("code")
        if code == 1 and "reduce the amount of data" in err.get("message", ""):
            raise TooMuchData(err["message"])
        if code in RATE_LIMIT_CODES or resp.status_code >= 500 or err.get("is_transient"):
            wait = max(60, regain_seconds(resp)) if code in RATE_LIMIT_CODES else 2 ** attempt * 5
            log(f"  API {code} ({err.get('message', '')[:70]}), waiting {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"{path}: {err.get('code')} {err.get('message')}")
    raise RuntimeError(f"{path}: gave up after repeated rate limits / errors")


def regain_seconds(resp: requests.Response) -> int:
    """Longest estimated_time_to_regain_access (minutes) in the usage headers, as seconds."""
    minutes = 0
    try:
        for entries in json.loads(resp.headers.get("x-business-use-case-usage", "{}")).values():
            for entry in entries:
                minutes = max(minutes, entry.get("estimated_time_to_regain_access", 0))
    except (ValueError, AttributeError):
        pass
    return minutes * 60


def paginate(path: str, params: dict, limit: int = 100) -> list[dict]:
    """All pages of an edge, halving the page size when Meta says it's too much data."""
    params = dict(params, limit=limit)
    out = []
    while True:
        try:
            body = call("GET", path, params)
        except TooMuchData:
            if params["limit"] <= 5:
                raise
            params["limit"] //= 2
            continue
        out += body.get("data", [])
        paging = body.get("paging", {})
        after = paging.get("cursors", {}).get("after")
        if not paging.get("next") or not after:
            return out
        params["after"] = after


def insights_report(account: str, params: dict) -> list[dict]:
    """Run an async insights job (needed for big date ranges) and return its rows.

    Jobs fail outright when the app hits Meta's request limit, so those are
    resubmitted after a growing wait instead of giving up.
    """
    for attempt in range(1, 9):
        run_id = call("POST", f"act_{account}/insights", params)["report_run_id"]
        while True:
            time.sleep(15)  # polls count against the rate limit too
            status = call("GET", run_id, {"fields": "async_status,async_percent_completion,error_code,error_message"})
            if status["async_status"] in ("Job Completed", "Job Failed", "Job Skipped"):
                break
        if status["async_status"] == "Job Completed":
            return paginate(f"{run_id}/insights", {}, limit=500)
        if status.get("error_code") not in RATE_LIMIT_CODES:
            raise RuntimeError(f"insights job {run_id} for act_{account}: {status['async_status']} "
                               f"{status.get('error_message', '')}")
        wait = 300 * attempt
        log(f"  job hit the app rate limit at {status.get('async_percent_completion')}%, "
            f"resubmitting in {wait // 60} min (attempt {attempt})")
        time.sleep(wait)
    raise RuntimeError(f"insights for act_{account}: still rate limited after 8 attempts")


def cached(account: str, name: str, refresh: bool, fetch) -> list | dict:
    path = RAW_DIR / account / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def fetch_account(account: str, refresh: bool) -> dict:
    status_filter = json.dumps([{"field": "ad.effective_status", "operator": "IN", "value": ALL_STATUSES}])
    log(f"act_{account}: lifetime insights")
    lifetime = cached(account, "insights_lifetime", refresh, lambda: insights_report(account, {
        "level": "ad", "date_preset": "maximum", "fields": ",".join(INSIGHT_FIELDS),
        "filtering": status_filter,
    }))
    log(f"act_{account}: daily insights")
    daily = cached(account, "insights_daily", refresh, lambda: insights_report(account, {
        "level": "ad", "date_preset": "maximum", "time_increment": 1, "fields": ",".join(DAILY_FIELDS),
        "filtering": status_filter,
    }))
    log(f"act_{account}: ads and creatives")
    ads = cached(account, "ads", refresh, lambda: paginate(f"act_{account}/ads", {
        "fields": AD_FIELDS,
        "filtering": json.dumps([{"field": "effective_status", "operator": "IN", "value": ALL_STATUSES}]),
    }))
    log(f"act_{account}: video library")
    library = cached(account, "advideos", refresh, lambda: paginate(
        f"act_{account}/advideos", {"fields": "id,title,length,source"}))
    log(f"act_{account}: {len(lifetime)} ads with delivery, {len(daily)} ad-days, "
        f"{len(ads)} ads, {len(library)} library videos")
    return {"lifetime": lifetime, "daily": daily, "ads": ads, "library": library}


def page_tokens(accounts: list[dict]) -> dict[str, str]:
    """Page access tokens for pages the token's system user is assigned to (may be none)."""
    tokens = {}
    businesses = {a["business"]["id"] for a in accounts if a.get("business")}
    for business in businesses:
        for edge in ("owned_pages", "client_pages"):
            try:
                pages = paginate(f"{business}/{edge}", {"fields": "id,name,access_token"})
            except RuntimeError:
                continue
            tokens.update({p["id"]: p["access_token"] for p in pages if p.get("access_token")})
    return tokens


def page_videos(ids_by_page: dict[str, set[str]], tokens: dict[str, str]) -> dict[str, dict]:
    """Title/length/source for videos owned by pages we have a page token for."""
    videos = {}
    for page_id, ids in ids_by_page.items():
        token = tokens.get(page_id)
        if not token:
            continue
        ids = sorted(ids)
        for i in range(0, len(ids), 50):
            try:
                body = session.get(f"{API}/", params={"ids": ",".join(ids[i:i + 50]),
                                                      "fields": "id,title,length,source",
                                                      "access_token": token}, timeout=120).json()
            except requests.RequestException:
                continue
            if "error" not in body:
                videos.update(body)
    return videos


def actions_map(items: list[dict] | None) -> dict[str, float]:
    return {a["action_type"]: float(a["value"]) for a in items or []}


def first(values: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in values:
            return values[key]
    return None


def ratio(num, den) -> float | None:
    return round(num / den, 6) if num is not None and den else None


def ad_videos(ad: dict) -> tuple[list[str], str | None]:
    creative = ad.get("creative") or {}
    spec = creative.get("object_story_spec") or {}
    ids = [creative.get("video_id") or (spec.get("video_data") or {}).get("video_id")]
    ids += [v.get("video_id") for v in (creative.get("asset_feed_spec") or {}).get("videos", [])]
    return list(dict.fromkeys(v for v in ids if v)), spec.get("page_id")


def build_rows(account: dict, data: dict, videos: dict[str, dict]) -> list[dict]:
    ads = {a["id"]: a for a in data["ads"]}
    days = defaultdict(list)
    for d in data["daily"]:
        if float(d.get("impressions") or 0) > 0:
            days[d["ad_id"]].append(d["date_start"])

    rows = []
    for ins in data["lifetime"]:
        ad = ads.get(ins["ad_id"], {})
        creative = ad.get("creative") or {}
        video_ids, _ = ad_videos(ad)
        vinfo = [videos.get(v, {}) for v in video_ids]
        act = actions_map(ins.get("actions"))
        values = actions_map(ins.get("action_values"))
        spend = float(ins.get("spend") or 0)
        impressions = float(ins.get("impressions") or 0)
        views_3s = act.get("video_view")
        thruplays = first(actions_map(ins.get("video_thruplay_watched_actions")), "video_view")
        purchases = first(act, "omni_purchase", "purchase", "offsite_conversion.fb_pixel_purchase")
        leads = first(act, "lead", "offsite_conversion.fb_pixel_lead")
        ad_days = sorted(days.get(ins["ad_id"], []))
        rows.append({
            "account_id": account["account_id"],
            "account_name": ins.get("account_name") or account.get("name"),
            "currency": account.get("currency"),
            "campaign_id": ins.get("campaign_id"),
            "campaign_name": ins.get("campaign_name"),
            "objective": ins.get("objective"),
            "adset_id": ins.get("adset_id"),
            "adset_name": ins.get("adset_name"),
            "ad_id": ins["ad_id"],
            "ad_name": ins.get("ad_name"),
            "ad_status": ad.get("effective_status"),
            "ad_created": (ad.get("created_time") or "")[:10],
            "first_day": ad_days[0] if ad_days else None,
            "last_day": ad_days[-1] if ad_days else None,
            "days_advertised": len(ad_days),
            "video_id": "|".join(video_ids),
            "video_file_name": "|".join(v.get("title") or "" for v in vinfo).strip("|"),
            "video_length_s": "|".join(str(v.get("length") or "") for v in vinfo).strip("|"),
            "download_link": "|".join(v.get("source") or "" for v in vinfo).strip("|"),
            "watch_link": "|".join(f"https://www.facebook.com/watch/?v={v}" for v in video_ids),
            "thumbnail_url": creative.get("thumbnail_url"),
            "spend": spend,
            "impressions": int(impressions),
            "reach": ins.get("reach"),
            "frequency": ins.get("frequency"),
            "cpm": ins.get("cpm"),
            "clicks": ins.get("clicks"),
            "ctr": ins.get("ctr"),
            "link_clicks": ins.get("inline_link_clicks"),
            "link_ctr": ins.get("inline_link_click_ctr"),
            "cpc": ins.get("cpc"),
            "video_3s_views": views_3s,
            "hook_rate": ratio(views_3s, impressions),
            "thruplays": thruplays,
            "hold_rate": ratio(thruplays, views_3s),
            "p25_views": first(actions_map(ins.get("video_p25_watched_actions")), "video_view"),
            "p50_views": first(actions_map(ins.get("video_p50_watched_actions")), "video_view"),
            "p75_views": first(actions_map(ins.get("video_p75_watched_actions")), "video_view"),
            "p100_views": first(actions_map(ins.get("video_p100_watched_actions")), "video_view"),
            "avg_watch_s": first(actions_map(ins.get("video_avg_time_watched_actions")), "video_view"),
            "landing_page_views": first(act, "omni_landing_page_view", "landing_page_view"),
            "add_to_cart": first(act, "omni_add_to_cart", "add_to_cart"),
            "initiate_checkout": first(act, "omni_initiated_checkout", "initiate_checkout"),
            "purchases": purchases,
            "purchase_value": first(values, "omni_purchase", "purchase", "offsite_conversion.fb_pixel_purchase"),
            "roas": first(actions_map(ins.get("purchase_roas")), "omni_purchase", "purchase"),
            "cost_per_purchase": ratio(spend, purchases),
            "leads": leads,
            "cost_per_lead": ratio(spend, leads),
            "messaging_started": act.get("onsite_conversion.messaging_conversation_started_7d"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounts", nargs="*", help="ad account ids (default: every account the token sees)")
    parser.add_argument("--refresh", action="store_true", help="refetch instead of using cached API responses")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("META_ACCESS_TOKEN"):
        sys.exit("Set META_ACCESS_TOKEN in .env")

    accounts = paginate("me/adaccounts", {"fields": "account_id,name,currency,business{id,name}"})
    if args.accounts:
        accounts = [a for a in accounts if a["account_id"] in set(args.accounts)]
    log(f"{len(accounts)} ad accounts: {', '.join(a['name'] for a in accounts)}")

    data = {}
    for account in accounts:
        try:
            data[account["account_id"]] = fetch_account(account["account_id"], args.refresh)
        except Exception as exc:  # one broken account shouldn't lose the rest
            log(f"act_{account['account_id']}: FAILED, skipped: {exc}")

    # Video details: the ad account libraries first, then page-owned videos if we have page tokens.
    videos = {v["id"]: v for d in data.values() for v in d["library"]}
    ids_by_page = defaultdict(set)
    for d in data.values():
        for ad in d["ads"]:
            ids, page_id = ad_videos(ad)
            for v in ids:
                if v not in videos and page_id:
                    ids_by_page[page_id].add(v)
    tokens = page_tokens(accounts)
    log(f"page tokens for {len(tokens)} pages; {sum(map(len, ids_by_page.values()))} ad videos not in a library")
    videos.update(page_videos(ids_by_page, tokens))

    rows, daily_rows = [], []
    for account in accounts:
        d = data.get(account["account_id"])
        if not d:
            continue
        rows += build_rows(account, d, videos)
        daily_rows += [{"account_id": account["account_id"], "date": x["date_start"], **{k: x.get(k) for k in DAILY_FIELDS}}
                       for x in d["daily"]]
    rows.sort(key=lambda r: (r["account_id"], -r["spend"]))

    stamp = run_stamp()
    out_csv = META_DIR / f"meta_ads_{stamp}.csv"
    daily_csv = META_DIR / f"meta_ads_daily_{stamp}.csv"
    META_DIR.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows Malay text/emoji
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with daily_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["account_id", "date", *DAILY_FIELDS])
        writer.writeheader()
        writer.writerows(daily_rows)

    with_video = sum(1 for r in rows if r["video_id"])
    with_link = sum(1 for r in rows if r["download_link"])
    log(f"\nWrote {len(rows)} ads to {out_csv} (total spend {sum(r['spend'] for r in rows):,.2f})")
    log(f"  {with_video} are video ads, {with_link} have a download link")
    log(f"Wrote {len(daily_rows)} ad-days to {daily_csv}")


if __name__ == "__main__":
    main()
