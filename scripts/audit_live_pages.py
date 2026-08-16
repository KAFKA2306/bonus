#!/usr/bin/env python3
"""Verify deployed Pages bytes against the generated local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from bonus_hypotheses import latest_hypothesis
from company_estimates import latest_company_estimation_model
from generate_verified_bonus_summary import DATA_DIR, latest_snapshot
from quantitative_benchmarks import latest_quantitative_benchmarks
from source_survey import latest_source_survey

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DEFAULT_URL = "https://kafka2306.github.io/bonus/"


def fetch_once(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "KAFKA2306-bonus-live-audit/2.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        return response.read()


def describe(body: bytes) -> str:
    preview = body[:800].decode("utf-8", errors="replace").replace("\n", " ")
    return (
        f"bytes={len(body)} sha256={hashlib.sha256(body).hexdigest()} "
        f"preview={preview!r}"
    )


def cache_busted(url: str, token: str, attempt: int) -> str:
    separator = "&" if urllib.parse.urlsplit(url).query else "?"
    return f"{url}{separator}audit={token}&attempt={attempt}"


def fetch_until(
    url: str,
    predicate: Callable[[bytes], bool],
    attempts: int,
    delay: float,
    token: str,
    expectation: str,
) -> bytes:
    last_body = b""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            last_body = fetch_once(cache_busted(url, token, attempt))
            if predicate(last_body):
                return last_body
            last_error = RuntimeError(
                f"response did not satisfy {expectation}: {describe(last_body)}"
            )
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(delay)
    raise RuntimeError(
        f"failed after {attempts} attempts: {url}; expected {expectation}; "
        f"last_error={last_error}; last_body={describe(last_body)}"
    )


def _relative_latest(selector, data_dir: Path, root: Path) -> str:
    snapshot = selector(data_dir).resolve()
    try:
        return str(snapshot.relative_to(root.resolve()))
    except ValueError:
        return str(snapshot)


def expected_generated_from(data_dir: Path = DATA_DIR, root: Path = ROOT) -> str:
    return _relative_latest(latest_snapshot, data_dir, root)


def expected_source_survey_from(data_dir: Path = DATA_DIR, root: Path = ROOT) -> str:
    return _relative_latest(latest_source_survey, data_dir, root)


def expected_quantitative_benchmarks_from(
    data_dir: Path = DATA_DIR, root: Path = ROOT
) -> str:
    return _relative_latest(latest_quantitative_benchmarks, data_dir, root)


def expected_hypotheses_from(data_dir: Path = DATA_DIR, root: Path = ROOT) -> str:
    return _relative_latest(latest_hypothesis, data_dir, root)


def expected_company_model_from(data_dir: Path = DATA_DIR, root: Path = ROOT) -> str:
    return _relative_latest(latest_company_estimation_model, data_dir, root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.url.rstrip("/") + "/"
    expected_paths = {
        "": DOCS / "index.html",
        "styles.css": DOCS / "styles.css",
        "app.js": DOCS / "app.js",
        "data/bonus.json": DOCS / "data" / "bonus.json",
    }
    token = hashlib.sha256(
        b"".join(path.read_bytes() for path in expected_paths.values())
    ).hexdigest()[:16]

    for relative_url, local_path in expected_paths.items():
        if not local_path.exists():
            raise SystemExit(f"local generated artifact is missing: {local_path}")
        expected = local_path.read_bytes()
        fetch_until(
            base + relative_url,
            lambda body, expected=expected: body == expected,
            args.attempts,
            args.delay,
            token,
            f"byte equality with {relative_url or 'index.html'}",
        )

    public = json.loads((DOCS / "data" / "bonus.json").read_text(encoding="utf-8"))
    if public.get("generated_from") != expected_generated_from():
        raise SystemExit("public JSON is not from latest verified facts")
    if public.get("source_survey_generated_from") != expected_source_survey_from():
        raise SystemExit("public JSON is not from latest source survey")
    if public.get("quantitative_benchmarks_generated_from") != expected_quantitative_benchmarks_from():
        raise SystemExit("public JSON is not from latest quantitative benchmarks")
    if public.get("hypotheses_generated_from") != expected_hypotheses_from():
        raise SystemExit("public JSON is not from latest company priors")
    if public.get("company_estimation_model_generated_from") != expected_company_model_from():
        raise SystemExit("public JSON is not from latest company estimation model")
    summary = public.get("summary", {})
    if summary.get("record_count") != 30:
        raise SystemExit("public JSON does not contain all 30 companies")
    if summary.get("quantified_company_count") != 30:
        raise SystemExit("public JSON does not quantify all 30 companies")
    if summary.get("quantitative_benchmark_count") != 11:
        raise SystemExit("public JSON does not contain all public quantitative benchmarks")
    if not summary.get("median_estimated_months") or not summary.get("median_estimated_amount_yen"):
        raise SystemExit("public JSON estimate summary is incomplete")
    if len(public.get("sector_anchors", [])) != 6:
        raise SystemExit("public JSON does not contain all six sector anchors")
    if public.get("universe", {}).get("coverage_ratio") != 1.0:
        raise SystemExit("public JSON coverage is not 100%")
    if any("estimate" not in item for item in public.get("records", [])):
        raise SystemExit("one or more company records lack an estimate")
    print(
        "PASS: live dashboard returned HTTP 200 and index/CSS/JS/JSON exactly match "
        "the generated 30-company, 6-sector and 11-benchmark build"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
