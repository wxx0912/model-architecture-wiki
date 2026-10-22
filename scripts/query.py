#!/usr/bin/env python3
"""Deterministic search over page metadata and body text."""
from __future__ import annotations

import argparse
import re
import sys

from wiki_utils import page_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--tag")
    parser.add_argument("--type", choices=["paradigm", "component", "model", "check"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    terms = [t.lower() for t in re.findall(r"[^\s,]+", args.query)]
    rows = []
    for path, meta, body in page_records():
        tags = [str(t).lower() for t in meta.get("tags", [])]
        haystack = " ".join([str(meta.get("title", "")), str(meta.get("id", "")), " ".join(tags), body]).lower()
        if args.tag and args.tag.lower() not in tags:
            continue
        if args.type and meta.get("type") != args.type:
            continue
        if terms and not all(term in haystack for term in terms):
            continue
        score = sum(haystack.count(term) for term in terms) if terms else 0
        rows.append((score, str(meta.get("id", "")), str(meta.get("title", "")), path))
    rows.sort(key=lambda row: (-row[0], row[1]))
    for _, page_id, title, path in rows[: max(args.limit, 0)]:
        print(f"{page_id}\t{title}\t{path.relative_to(path.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
