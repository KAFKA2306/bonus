#!/usr/bin/env python3
"""Verify public GitHub Pages bytes against the published gh-pages branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

DEFAULT_URL = "https://kafka2306.github.io/bonus/"
DEFAULT_PUBLISHED_URL = "https://raw.githubusercontent.com/KAFKA2306/bonus/gh-pages/"
PUBLISHED_PATHS = ("index.html", "styles.css", "app.js", "data/bonus.json")


def fetch_once(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "KAFKA2306-bonus-live-audit/3.0",
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


def validate_payload(public: dict) -> None:
    summary = public.get("summary", {})
    if summary.get("record_count") != 225:
        raise RuntimeError("public JSON does not contain all 225 Nikkei 225 companies")
    if summary.get("quantified_company_count") != 225:
        raise RuntimeError("public JSON does not quantify all 225 Nikkei 225 companies")
    if summary.get("quantitative_benchmark_count") != 11:
        raise RuntimeError("public JSON does not contain all public quantitative benchmarks")
    if not summary.get("median_estimated_months"):
        raise RuntimeError("public JSON estimated-month summary is incomplete")
    if len(public.get("sector_anchors", [])) != 6:
        raise RuntimeError("public JSON does not contain all six sector anchors")
    universe = public.get("universe", {})
    if universe.get("tracked_companies") != 225 or universe.get("covered_companies") != 225:
        raise RuntimeError("public JSON universe does not contain all 225 companies")
    if universe.get("coverage_ratio") != 1.0:
        raise RuntimeError("public JSON coverage is not 100%")
    if any("estimate" not in item for item in public.get("records", [])):
        raise RuntimeError("one or more company records lack an estimate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--published-url", default=DEFAULT_PUBLISHED_URL)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    public_base = args.url.rstrip("/") + "/"
    published_base = args.published_url.rstrip("/") + "/"
    published = {
        path: fetch_once(published_base + path)
        for path in PUBLISHED_PATHS
    }
    token = hashlib.sha256(b"".join(published.values())).hexdigest()[:16]

    for path, expected in published.items():
        fetch_until(
            public_base + path,
            lambda body, expected=expected: body == expected,
            args.attempts,
            args.delay,
            token,
            f"byte equality with published {path}",
        )

    validate_payload(json.loads(published["data/bonus.json"]))
    print(
        "PASS: public Pages bytes match gh-pages for index/CSS/JS/JSON and the "
        "published payload contains 225 companies, 6 sector anchors and 11 benchmarks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
