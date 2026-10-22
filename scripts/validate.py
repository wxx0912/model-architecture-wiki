#!/usr/bin/env python3
"""Validate page frontmatter, references, tags, and Markdown links."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from datetime import date

from wiki_utils import ROOT, page_records, source_entries, source_ids

REQUIRED = {"id", "title", "type", "tags", "related", "sources", "confidence", "evidence_scope", "validated_at"}
TYPES = {"paradigm", "component", "model", "check"}
CONFIDENCE = {"verified", "source-reported", "inferred", "experimental"}
EVIDENCE_SCOPE = {"source-reported", "inferred", "experimental", "mixed"}


def controlled_tags() -> set[str]:
    tags = set()
    for line in (ROOT / "data" / "tags.yaml").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*-\s*([^#\s]+)", line)
        if match:
            tags.add(match.group(1).strip("'\""))
    return tags


def markdown_links(path: Path, body: str):
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        yield target.split("#", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-external", action="store_true", help="reserved for a network-enabled CI job")
    args = parser.parse_args()
    del args
    errors: list[str] = []
    warnings: list[str] = []
    known_tags = controlled_tags()
    known_sources = source_ids()
    sources = source_entries()
    if len(sources) != len(known_sources):
        errors.append("source registry contains malformed or duplicate rows")
    for source_id, (_, url, purpose) in sources.items():
        if not re.fullmatch(r"https?://[^\s|]+", url):
            errors.append(f"source {source_id}: invalid URL {url!r}")
        if not purpose:
            errors.append(f"source {source_id}: missing purpose")
    records = list(page_records())
    ids: dict[str, Path] = {}
    for path, meta, body in records:
        label = str(path.relative_to(ROOT))
        missing = REQUIRED - set(meta)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        page_id = meta.get("id")
        if not isinstance(page_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", page_id):
            errors.append(f"{label}: invalid id {page_id!r}")
        elif page_id in ids:
            errors.append(f"duplicate id {page_id}: {ids[page_id]} and {label}")
        else:
            ids[page_id] = path
        if meta.get("type") not in TYPES:
            errors.append(f"{label}: invalid type {meta.get('type')!r}")
        if meta.get("confidence") not in CONFIDENCE:
            errors.append(f"{label}: invalid confidence {meta.get('confidence')!r}")
        if meta.get("evidence_scope") not in EVIDENCE_SCOPE:
            errors.append(f"{label}: invalid evidence_scope {meta.get('evidence_scope')!r}")
        validated_at = meta.get("validated_at")
        if not isinstance(validated_at, str):
            errors.append(f"{label}: validated_at must be YYYY-MM-DD")
        else:
            try:
                date.fromisoformat(validated_at)
            except ValueError:
                errors.append(f"{label}: invalid validated_at {validated_at!r}")
        if not isinstance(meta.get("tags"), list):
            errors.append(f"{label}: tags must be a list")
        else:
            for tag in meta["tags"]:
                if tag not in known_tags:
                    errors.append(f"{label}: unknown tag {tag!r}")
        for field in ("related", "sources"):
            if not isinstance(meta.get(field), list):
                errors.append(f"{label}: {field} must be a list")
        for related in meta.get("related", []) if isinstance(meta.get("related"), list) else []:
            if related not in ids and not any(str(other.get("id")) == related for _, other, _ in records):
                errors.append(f"{label}: unknown related page {related!r}")
        for source in meta.get("sources", []) if isinstance(meta.get("sources"), list) else []:
            if source not in known_sources:
                errors.append(f"{label}: unknown source {source!r}")
        for target in markdown_links(path, body):
            target_path = (path.parent / target).resolve()
            if target and not target_path.exists():
                errors.append(f"{label}: broken relative link {target!r}")
    # Validate non-page markdown links in README/index/contribution docs too.
    for path in [ROOT / "README.md", ROOT / "SKILL.md", ROOT / "index.md", ROOT / "index.generated.md", ROOT / "CONTRIBUTING.md"]:
        if not path.exists():
            continue
        for target in markdown_links(path, path.read_text(encoding="utf-8")):
            if target and not (path.parent / target).resolve().exists():
                errors.append(f"{path.relative_to(ROOT)}: broken relative link {target!r}")
    for warning in warnings:
        print(f"WARN: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"validation failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"validation passed: {len(records)} pages, {len(known_sources)} sources, {len(known_tags)} tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
