#!/usr/bin/env python3
"""Verify deployed Pages bytes against locally generated fact and hypothesis artifacts."""

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
from generate_verified_bonus_summary import DATA_DIR, latest_snapshot

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DEFAULT_URL = "https://kafka2306.github.io/bonus/"
BUILD_MARKER = b'name="bonus-build" content="verified-pages-v4"'
EXPECTED_TITLE = "主要30社 賞与制度・推定比較表".encode("utf-8")


def fetch_once(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "KAFKA2306-bonus-live-audit/1.0",
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
    digest = hashlib.sha256(body).hexdigest()
    return f"bytes={len(body)} sha256={digest} preview={preview!r}"


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


def expected_generated_from(
    data_dir: Path = DATA_DIR,
    root: Path = ROOT,
) -> str:
    return _relative_latest(latest_snapshot, data_dir, root)


def expected_hypotheses_from(
    data_dir: Path = DATA_DIR,
    root: Path = ROOT,
) -> str:
    return _relative_latest(latest_hypothesis, data_dir, root)


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
        "styles.css": DOCS / "styles.css",
        "app.js": DOCS / "app.js",
        "data/bonus.json": DOCS / "data" / "bonus.json",
    }
    token = hashlib.sha256(
        b"".join(path.read_bytes() for path in expected_paths.values())
    ).hexdigest()[:16]

    index = fetch_until(
        base,
        lambda body: BUILD_MARKER in body and EXPECTED_TITLE in body,
        args.attempts,
        args.delay,
        token,
        "verified-pages-v4 marker and Japanese comparison-table title",
    )
    if BUILD_MARKER not in index:
        raise SystemExit("live index.html is missing the v4 build marker")
    if EXPECTED_TITLE not in index:
        raise SystemExit("live index.html is missing the expected Japanese title")

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
            f"byte equality with {relative_url}",
        )

    public = json.loads((DOCS / "data" / "bonus.json").read_text(encoding="utf-8"))
    if public.get("schema_version") != 1:
        raise SystemExit("unexpected public JSON schema version")

    expected_source = expected_generated_from()
    if public.get("generated_from") != expected_source:
        raise SystemExit(
            "public JSON was not generated from the latest verified snapshot: "
            f"expected={expected_source!r} actual={public.get('generated_from')!r}"
        )

    expected_hypotheses = expected_hypotheses_from()
    if public.get("hypotheses_generated_from") != expected_hypotheses:
        raise SystemExit(
            "public JSON was not generated from the latest hypothesis snapshot: "
            f"expected={expected_hypotheses!r} "
            f"actual={public.get('hypotheses_generated_from')!r}"
        )
    summary = public.get("summary", {})
    universe = public.get("universe", {})
    if summary.get("record_count") != 30 or summary.get("hypothesis_count") != 30:
        raise SystemExit("public JSON does not contain all 30 companies")
    if universe.get("coverage_ratio") != 1.0:
        raise SystemExit("public JSON coverage is not 100%")

    print(
        "PASS: live comparison table returned HTTP 200 and index/CSS/JS/JSON "
        "exactly match the 30-company fact and hypothesis build"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
