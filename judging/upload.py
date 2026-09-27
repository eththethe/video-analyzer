"""Bulk-upload vid/*.mp4 to GCS so gemini.py can reuse them without uploading.

Objects land at gs://$GCS_BUCKET/video-analyzer/<file name>, the same path
gemini.py's upload_video() checks, so judging runs skip the upload step.

Setup (once):
    pip install google-cloud-storage
    gcloud auth application-default login

Usage:
    python judging/upload.py                  # upload everything in vid/
    python judging/upload.py vid/some.mp4     # upload specific files
    python judging/upload.py --dry-run        # show what would be uploaded
    python judging/upload.py --workers 8      # parallel uploads (default 4)
    python judging/upload.py --force          # re-upload even if already there

A file is skipped when an object with the same name and byte size already
exists, so the script is safe to re-run after a failure or interruption.
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY

from gemini import GCS_PREFIX, VIDEO_DIR, load_dotenv

CHUNK_SIZE = 32 * 1024 * 1024  # resumable upload chunk; must be a multiple of 256 KiB

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def upload_one(bucket: storage.Bucket, path: Path) -> None:
    """Upload path via a chunked resumable upload."""
    blob = bucket.blob(f"{GCS_PREFIX}/{path.name}", chunk_size=CHUNK_SIZE)
    blob.upload_from_filename(str(path), content_type="video/mp4", timeout=900, retry=DEFAULT_RETRY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="videos to upload (default: all of vid/)")
    parser.add_argument("--workers", type=int, default=4, help="parallel uploads (default 4)")
    parser.add_argument("--force", action="store_true", help="re-upload files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = parser.parse_args()

    load_dotenv()
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        sys.exit("Set GCS_BUCKET in .env")

    videos = args.paths or sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        sys.exit(f"No videos found in {VIDEO_DIR}")

    client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    bucket = client.bucket(bucket_name)

    # One listing call instead of one exists() call per file.
    existing = {Path(b.name).name: b.size for b in client.list_blobs(bucket, prefix=f"{GCS_PREFIX}/")}

    todo, skipped = [], 0
    for path in videos:
        if not args.force and existing.get(path.name) == path.stat().st_size:
            skipped += 1
        else:
            todo.append(path)

    total_bytes = sum(p.stat().st_size for p in todo)
    print(f"gs://{bucket_name}/{GCS_PREFIX}/  {len(videos)} videos: "
          f"{skipped} already uploaded, {len(todo)} to upload ({fmt_size(total_bytes)})")

    if args.dry_run:
        for path in todo:
            print(f"  would upload {path.name} ({fmt_size(path.stat().st_size)})")
        return
    if not todo:
        return

    start = time.monotonic()
    done_bytes, failed = 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(upload_one, bucket, path): path for path in todo}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            size = path.stat().st_size
            try:
                future.result()
                done_bytes += size
                rate = done_bytes / max(time.monotonic() - start, 1e-6)
                log(f"[{i}/{len(todo)}] ok    {path.name} ({fmt_size(size)}, {fmt_size(rate)}/s avg)")
            except Exception as exc:  # keep going; report at the end
                failed.append(path)
                log(f"[{i}/{len(todo)}] FAIL  {path.name}: {exc}")

    elapsed = time.monotonic() - start
    print(f"\nUploaded {len(todo) - len(failed)}/{len(todo)} in {elapsed / 60:.1f} min "
          f"({fmt_size(done_bytes)}).")
    if failed:
        print("Failed (re-run the script to retry):")
        for path in failed:
            print(f"  {path.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
