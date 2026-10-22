#!/usr/bin/env python3
"""Print one page selected by its stable ID."""
from __future__ import annotations

import argparse
import sys

from wiki_utils import page_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page_id")
    args = parser.parse_args()
    for path, meta, _ in page_records():
        if meta.get("id") == args.page_id:
            print(path.read_text(encoding="utf-8"), end="")
            return 0
    print(f"page not found: {args.page_id}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
