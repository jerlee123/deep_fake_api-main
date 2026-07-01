import argparse
import csv
import glob
import os
import re
import sys
from typing import List, Tuple
import requests

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]
_FALLBACK_SORT_INDEX = 10**9


def find_videos(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, "**", "*")
    candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    return [p for p in candidates if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS]


def extract_index(fname: str) -> Tuple[int, str]:
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"(?i)VIDEO_(\d+)_HD_SHORT", base)
    if m:
        try:
            return int(m.group(1)), fname
        except ValueError:
            pass
    m = re.search(r"(?i)VIDEO[_-]?(\d+)\b", base)
    if m:
        try:
            return int(m.group(1)), fname
        except ValueError:
            pass
    m = re.search(r"(\d+)", base)
    if m:
        try:
            return int(m.group(1)), fname
        except ValueError:
            pass
    return _FALLBACK_SORT_INDEX, fname


def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return default


def run_benchmark(input_dir: str, base_url: str, out_csv: str, timeout: int):
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        sys.exit(1)

    files = find_videos(input_dir)
    if not files:
        print(f"[ERROR] No video files found in {input_dir}")
        sys.exit(1)

    files.sort(key=lambda p: (extract_index(os.path.basename(p))[0], os.path.basename(p).lower()))
    rows = []

    print(f"Found {len(files)} video(s). Sending to {base_url}/predict ...")
    for v in files:
        print(" -", os.path.relpath(v, input_dir))
    print()

    for i, path in enumerate(files, start=1):
        fname = os.path.basename(path)
        sort_idx, _ = extract_index(fname)
        csv_idx = "" if sort_idx >= _FALLBACK_SORT_INDEX else sort_idx

        print(f"[{i}/{len(files)}] Processing {fname} ...", end=" ")
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    f"{base_url}/predict/",
                    files={"file": (fname, f, "application/octet-stream")},
                    timeout=timeout,
                )
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                rows.append({
                    "sort_index": sort_idx,
                    "index": csv_idx,
                    "filename": fname,
                    "label": "ERROR",
                    "confidence": "",
                    "processing_time_sec": "",
                    "video_duration_sec": "",
                    "file_size_mb": "",
                    "error": f"HTTP {resp.status_code}",
                })
                continue
            data = resp.json()
        except Exception as e:
            print("FAILED")
            rows.append({
                "sort_index": sort_idx,
                "index": csv_idx,
                "filename": fname,
                "label": "ERROR",
                "confidence": "",
                "processing_time_sec": "",
                "video_duration_sec": "",
                "file_size_mb": "",
                "error": str(e),
            })
            continue

        if isinstance(data, dict) and "error" in data:
            print("ERROR from API")
            rows.append({
                "sort_index": sort_idx,
                "index": csv_idx,
                "filename": fname,
                "label": "ERROR",
                "confidence": "",
                "processing_time_sec": "",
                "video_duration_sec": "",
                "file_size_mb": "",
                "error": data.get("error", "unknown error"),
            })
            continue

        label = data.get("result", "Unknown")
        conf = data.get("confidence", None)
        proc = data.get("processing_time_sec", None)
        meta = data.get("metadata", {}) or {}
        duration_sec = to_float(meta.get("duration"), default="")

        try:
            file_size_bytes = int(meta.get("file_size")) if meta.get("file_size") is not None else None
            size_mb = round(file_size_bytes / (1024 * 1024), 3) if file_size_bytes is not None else ""
        except Exception:
            size_mb = ""

        print(f"OK → {label} ({conf}%)")
        rows.append({
            "sort_index": sort_idx,
            "index": csv_idx,
            "filename": fname,
            "label": label,
            "confidence": conf,
            "processing_time_sec": proc,
            "video_duration_sec": duration_sec,
            "file_size_mb": size_mb,
            "error": "",
        })

    rows.sort(key=lambda r: (r.get("sort_index", _FALLBACK_SORT_INDEX), r.get("filename", "").lower()))
    fieldnames = ["index", "filename", "label", "confidence", "processing_time_sec", "video_duration_sec", "file_size_mb", "error"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved CSV → {out_csv}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SHORT_VIDEO_HD datasets (FAKE or REAL) via FastAPI /predict endpoint and export CSV."
    )
    parser.add_argument("--api", default="http://localhost:8000", help="Base URL of your API")
    parser.add_argument("--dataset", choices=["FAKE_DATA_SET", "REAL_DATA_SET"], default="FAKE_DATA_SET",
                        help="Choose dataset folder inside SHORT_VIDEO_HD")
    parser.add_argument("--out", default="benchmark_results.csv", help="Output CSV filename")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout (sec)")
    args = parser.parse_args()

    input_dir = os.path.join("SHORT_VIDEO_HD", args.dataset)
    base_url = args.api.rstrip("/")
    run_benchmark(input_dir=input_dir, base_url=base_url, out_csv=args.out, timeout=args.timeout)


if __name__ == "__main__":
    main()

# python result.py --dataset FAKE_DATA_SET