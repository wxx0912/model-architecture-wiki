"""Small dependency-free helpers for the model architecture wiki."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = ROOT / "wiki"


def split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening frontmatter delimiter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError("missing closing frontmatter delimiter") from exc
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :])


def parse_value(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        # Frontmatter lists in this repository contain simple unquoted tokens.
        return [part.strip().strip("'\"") for part in inner.split(",")]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw[1:-1]
    return raw


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    front, body = split_frontmatter(text)
    metadata: dict[str, object] = {}
    for line_no, line in enumerate(front.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"frontmatter line {line_no} has no ':'")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            raise ValueError(f"invalid frontmatter key: {key!r}")
        metadata[key] = parse_value(raw)
    return metadata, body


def page_files():
    return sorted(p for p in PAGE_ROOT.rglob("*.md") if p.name != "TEMPLATE.md")


def page_id_from_path(path: Path) -> str:
    metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    return str(metadata.get("id", ""))


def source_ids() -> set[str]:
    ids: set[str] = set()
    table = (ROOT / "sources" / "README.md").read_text(encoding="utf-8")
    for line in table.splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if match:
            ids.add(match.group(1))
    return ids


def source_entries() -> dict[str, tuple[str, str, str]]:
    """Return source id -> (kind, URL, purpose) from the registry table."""
    entries: dict[str, tuple[str, str, str]] = {}
    table = (ROOT / "sources" / "README.md").read_text(encoding="utf-8")
    for line in table.splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|\s*([^|]+)\|\s*(https?://[^|]+)\|\s*(.*?)\s*\|\s*$", line)
        if match:
            entries[match.group(1)] = (match.group(2).strip(), match.group(3).strip(), match.group(4).strip())
    return entries


def page_records():
    for path in page_files():
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        yield path, metadata, body
