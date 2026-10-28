"""共享库：知识页与来源页的读取、校验、检索与索引渲染。

校验输出带规则编号，便于定位条款。词表取值来自 data/vocabulary.yaml 与 data/schemas.yaml，
共享语义（证据强度、来源类别等）在本文件声明，不写进任何一个仓库的数据文件。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import namedtuple
from datetime import date
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

CONFIDENCE = ("verified", "source-reported", "inferred", "experimental")
EVIDENCE_SCOPE = ("source-reported", "inferred", "experimental", "mixed")
SOURCE_CATEGORY = ("paper", "repo", "doc", "report")
READ_FORM = ("latex", "html", "pdf", "repo-tree")
VERSION_KIND = ("paper-version", "commit", "tag", "revision", "unknown")
ENUMS = {
    "confidence": CONFIDENCE,
    "evidence_scope": EVIDENCE_SCOPE,
    "source_category": SOURCE_CATEGORY,
    "read_form": READ_FORM,
    "version_kind": VERSION_KIND,
}
CATEGORY_DIR = {"paper": "papers", "repo": "repos",
                "doc": "docs", "report": "reports"}
SOURCE_SECTIONS = ("这份来源是什么", "已读定位与阅读形式", "关键事实", "不能推出的结论")
EXCERPT_START = "<!-- excerpt-start -->"
EXCERPT_END = "<!-- excerpt-end -->"
GENERATED_FILES = ("index.generated.md", "INDEX.generated.md")

Page = namedtuple("Page", "path meta body")
Source = namedtuple("Source", "path meta body sections excerpts")


def _json(text, label):
    def hook(pairs):
        keys = [key for key, _ in pairs]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(f"{label}: JSON 重复键 {repeated}")
        return dict(pairs)

    value = json.loads(text, object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: 必须是 JSON 对象")
    return value


def _frontmatter(path):
    parts = path.read_text(encoding="utf-8").split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"{path}: 缺少 frontmatter 分隔符")
    return _json(parts[1], path), parts[2].strip()


def _data(root, name):
    return _json((Path(root) / "data" / name).read_text(encoding="utf-8"), f"data/{name}")


def vocabulary(root=ROOT):
    return _data(root, "vocabulary.yaml")


def schemas(root=ROOT):
    return _data(root, "schemas.yaml")


def pages(root=ROOT):
    records = []
    for path in sorted((Path(root) / "wiki").rglob("*.md")):
        meta, body = _frontmatter(path)
        records.append(Page(path, meta, body))
    return records


def _sections(body):
    sections, title, lines = {}, None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if title is not None:
                sections[title] = "\n".join(lines).strip()
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None:
        sections[title] = "\n".join(lines).strip()
    return sections


def sources(root=ROOT):
    registry = {}
    for dirname in CATEGORY_DIR.values():
        for path in sorted((Path(root) / "sources" / dirname).glob("*.md")):
            meta, body = _frontmatter(path)
            source_id = meta.get("id", path.stem)
            if source_id in registry:
                raise ValueError(f"{path}: 重复来源 ID {source_id}")
            pattern = re.escape(EXCERPT_START) + r"\n(.*?)\n" + re.escape(EXCERPT_END)
            excerpts = re.findall(pattern, body, re.S)
            registry[source_id] = Source(path, meta, body, _sections(body), excerpts)
    return registry


def _normalize(text):
    """比较用归一化：折叠大小写，标点、分隔符与下划线转空格，压平空白。"""
    return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", text.casefold())).strip()


def _term_in(term, text):
    """名称/标签匹配：ASCII 关键词按词首对齐，避免 "llm" 命中 "dllmquant" 这类子串误命中；
    含非 ASCII（如中文）的关键词仍按子串匹配。"""
    if term.isascii():
        return any(token.startswith(term) for token in text.split())
    return term in text


def search(records, query="", tag=None, kind=None, limit=10):
    """先按名称、别名与标签检索；没有命中再退回正文，退回的条目为弱相关。

    返回 (score, page, weak) 列表；weak 为真表示整批结果都是正文提及。
    """
    terms = _normalize(query).split()
    for include_body in (False, True):
        rows = []
        for page in records:
            meta = page.meta
            if tag and tag not in meta["tags"]:
                continue
            if kind and kind != meta["type"]:
                continue
            names = [_normalize(name) for name in [meta["id"], meta["title"], *meta["aliases"]]]
            name_text = " ".join(names)
            tag_text = _normalize(" ".join(meta["tags"]))
            body_text = _normalize(page.body) if include_body else ""
            hits = [(term, _term_in(term, name_text), _term_in(term, tag_text), _term_in(term, body_text))
                    for term in terms]
            if not all(name or tag or body for _, name, tag, body in hits):
                continue
            exact = bool(query) and _normalize(query) in names
            score = 100 * exact + sum(10 * name + 3 * tag + (1 if (name or tag or body) else 0)
                                      for _, name, tag, body in hits)
            rows.append((score, page, include_body))
        if rows or include_body:
            rows.sort(key=lambda row: (-row[0], row[1].meta["id"]))
            return rows[:limit]
    return []


def _field_errors(spec, meta, vocab, label, rules):
    fields_rule, value_rule, array_rule, date_rule = rules
    errors = []
    required = set(spec["required"])
    missing = sorted(required - set(meta))
    if missing:
        errors.append(f"{fields_rule} {label}: 缺少必需字段 {missing}")
    extra = sorted(set(meta) - required - set(spec.get("optional", [])))
    if extra:
        errors.append(f"{fields_rule} {label}: 表外字段 {extra}")
    for field, value in meta.items():
        rule = spec.get("rules", {}).get(field, {"type": "string"})
        kind = rule["type"]
        if value is None and rule.get("null"):
            continue
        if kind == "boolean":
            if not isinstance(value, bool):
                errors.append(f"{value_rule} {label}: {field} 必须为布尔值")
            continue
        if kind == "date":
            valid = isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            if valid:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    valid = False
            if not valid:
                errors.append(f"{date_rule} {label}: {field} 日期非法 {value!r}")
            continue
        if kind == "array":
            if (not isinstance(value, list) or (rule.get("non_empty") and not value)
                    or not all(isinstance(item, str) and item for item in value)):
                errors.append(f"{array_rule} {label}: {field} 必须为非空字符串数组")
                continue
            if len(set(value)) != len(value):
                errors.append(f"{array_rule} {label}: {field} 存在重复项")
            values = value
        else:
            if not isinstance(value, str) or not value:
                errors.append(f"{value_rule} {label}: {field} 必须为非空字符串")
                continue
            values = [value]
        if "enum" in rule:
            unknown = [item for item in values if item not in ENUMS[rule["enum"]]]
            if unknown:
                errors.append(f"{value_rule} {label}: {field} 取值非法 {unknown}")
        if "vocabulary" in rule:
            known = [item["name"] if isinstance(item, dict) else item
                     for item in vocab[rule["vocabulary"]]]
            unknown = [item for item in values if item not in known]
            if unknown:
                errors.append(f"{value_rule} {label}: {field} 不在词表中 {unknown}")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], values[0]):
            errors.append(f"{value_rule} {label}: {field} 形状非法 {values[0]!r}")
    return errors


def _anchors(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    anchors, counts = set(), {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.M):
        slug = re.sub(r"[^\w\-\s]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return anchors


def _link_errors(path, root):
    errors = []
    label = path.relative_to(root).as_posix()
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    for angle, plain in re.findall(r"\]\((?:<([^>]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\)", text):
        destination = angle or plain
        if re.match(r"[a-zA-Z]+:", destination):
            continue
        target, _, fragment = destination.partition("#")
        resolved = path.parent / unquote(target) if target else path
        if not resolved.exists():
            errors.append(f"P9 {label}: 失效本地链接 {destination}")
        elif fragment and resolved.suffix == ".md":
            if unquote(fragment) not in _anchors(resolved.read_text(encoding="utf-8")):
                errors.append(f"P9 {label}: 失效标题锚点 {destination}")
    return errors


def _excerpt_errors(source, label):
    starts = source.body.count(EXCERPT_START)
    ends = source.body.count(EXCERPT_END)
    digest = source.meta.get("excerpt_sha256")
    if not starts and not ends:
        if digest is not None:
            return [f"S6 {label}: 无逐字片段时 excerpt_sha256 必须为 null"]
        return []
    if starts != 1 or ends != 1 or len(source.excerpts) != 1:
        return [f"S6 {label}: 哨兵必须恰好成对出现一次，实际 start={starts} end={ends}"]
    actual = hashlib.sha256(source.excerpts[0].encode("utf-8")).hexdigest()
    if digest != actual:
        return [f"S6 {label}: excerpt_sha256 与片段不符，实际 {actual}"]
    return []


def validate(root=ROOT):
    """返回 (errors, warnings, 页面数, 来源数)。"""
    root = Path(root)
    vocab, spec = vocabulary(root), schemas(root)
    errors, warnings = [], []
    try:
        records = pages(root)
    except ValueError as exc:
        errors.append(f"P1 {exc}")
        records = []
    try:
        registry = sources(root)
    except ValueError as exc:
        errors.append(f"S9 {exc}")
        registry = {}

    ids = set()
    for page in records:
        label = page.path.relative_to(root).as_posix()
        meta = page.meta
        errors += _field_errors(spec["page"], meta, vocab, label, ("P2", "P4", "P5", "P6"))
        page_id = meta.get("id")
        if isinstance(page_id, str) and re.fullmatch(r"[a-z][a-z0-9-]+", page_id):
            if page_id in ids:
                errors.append(f"P3 {label}: 重复 ID {page_id}")
            ids.add(page_id)
        if not page.body.startswith("# "):
            errors.append(f"P7 {label}: 正文缺少一级标题")
        if not meta.get("sources"):
            errors.append(f"P8 {label}: sources 不得为空")
        aliases = [_normalize(item) for item in meta.get("aliases") or [] if isinstance(item, str)]
        repeated = sorted({item for item in aliases if aliases.count(item) > 1})
        if repeated:
            errors.append(f"P10 {label}: aliases 归一化后重复 {repeated}")
    for page in records:
        label = page.path.relative_to(root).as_posix()
        related = page.meta.get("related")
        for target in related if isinstance(related, list) else []:
            if isinstance(target, str) and target not in ids:
                errors.append(f"P8 {label}: related 未登记 {target}")

    referenced = {source_id for page in records
                  for source_id in page.meta.get("sources") or [] if isinstance(source_id, str)}
    for page in records:
        label = page.path.relative_to(root).as_posix()
        for source_id in page.meta.get("sources") or []:
            if isinstance(source_id, str) and source_id not in registry:
                errors.append(f"S8 {label}: sources 未登记 {source_id}")
    for source_id, source in registry.items():
        label = source.path.relative_to(root).as_posix()
        meta = source.meta
        errors += _field_errors(spec["source"], meta, vocab, label, ("S2", "S3", "S5", "S4"))
        errors += _excerpt_errors(source, label)
        if not re.fullmatch(r"https?://\S+", str(meta.get("url", ""))):
            errors.append(f"S7 {label}: url 必须以 http:// 或 https:// 开头")
        expected = CATEGORY_DIR.get(meta.get("category"))
        if expected and (source.path.parent.name != expected or source.path.stem != source_id):
            errors.append(f"S1 {label}: category={meta.get('category')} 应位于 sources/{expected}/{source_id}.md")
        missing = [name for name in SOURCE_SECTIONS if name not in source.sections]
        if missing:
            errors.append(f"S5 {label}: 缺少二级标题 {missing}")
        elif not source.sections["不能推出的结论"]:
            errors.append(f"S5 {label}: “不能推出的结论”不得为空")
        if source_id not in referenced:
            warnings.append(f"S10 {label}: 无页面引用")
    for path in sorted((root / "sources").rglob("*.md")):
        if path.name in GENERATED_FILES:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("| `"):
                errors.append(f"S11 {path.relative_to(root).as_posix()}:{number}: 来源目录下不得使用来源表格")

    # 每个标签要么归入某个索引分组，要么登记为跨切面总分类（cross_cutting_tags）；
    # 分组引用的标签必须真的被页面使用。
    grouped = {tag for group in (vocab.get("index_groups") or {}).get("groups") or []
               if isinstance(group, dict) for tag in group.get("tags") or []}
    loose = {tag for tag in vocab.get("cross_cutting_tags") or [] if isinstance(tag, str)}
    used = {tag for page in records for tag in page.meta.get("tags") or []}
    for tag in sorted(used - grouped - loose):
        errors.append(f"P12 页面使用的标签既不在索引分组也未登记为跨切面标签: {tag}")
    for tag in sorted(grouped - used):
        warnings.append(f"P11 索引分组引用的标签没有页面使用: {tag}")

    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        errors += _link_errors(path, root)
    return errors, warnings, len(records), len(registry)


def render_page_index(root, records, vocab):
    lines = ["# 页面索引", "", "> 由 `python3 scripts/build_index.py` 生成，请勿手改。", ""]
    for page_type in vocab["page_types"]:
        lines += [f"## {page_type['label']}", ""]
        for page in sorted(records, key=lambda item: item.meta["id"]):
            if page.meta["type"] == page_type["name"]:
                target = page.path.relative_to(root).as_posix()
                lines.append(f"- `{page.meta['id']}` — [{page.meta['title']}]({target})")
        lines.append("")
    config = vocab.get("index_groups") or {}
    groups = config.get("groups") or []
    excluded = set(config.get("exclude_types") or [])
    if groups:
        title = config.get("title") or "按分类检索"
        lines += [f"## {title}", "",
                  "> 按页面标签聚合，一个页面可出现在多组；完整标签索引见下节。", ""]
        for group in groups:
            if not isinstance(group, dict):
                continue
            tags = [tag for tag in group.get("tags") or [] if isinstance(tag, str)]
            matched = sorted((page for page in records
                              if page.meta["type"] not in excluded
                              and set(page.meta.get("tags") or []) & set(tags)),
                             key=lambda item: item.meta["id"])
            if not matched:
                continue
            links = "；".join(f"[{page.meta['title']}]({page.path.relative_to(root).as_posix()})"
                             for page in matched)
            lines.append(f"- **{group.get('label')}**（{'、'.join(tags)}）：{links}")
        lines.append("")
    lines += ["## 按机制标签", ""]
    for tag in sorted({tag for page in records for tag in page.meta.get("tags") or []}):
        links = [f"[{page.meta['title']}]({page.path.relative_to(root).as_posix()})"
                 for page in records if tag in (page.meta.get("tags") or [])]
        lines.append(f"- `{tag}`：" + "；".join(links))
    return "\n".join(lines) + "\n"


def render_source_index(root, registry, records, vocab):
    counts = {source_id: 0 for source_id in registry}
    for page in records:
        for source_id in page.meta.get("sources", []):
            if source_id in counts:
                counts[source_id] += 1
    lines = ["# 来源索引", "", "> 由 `python3 scripts/build_index.py` 生成，请勿手改。",
             "", "主题说明见 topics/。", ""]
    for topic in vocab["source_topics"]:
        lines += [f"## {topic['label']}", "",
                  f"主题说明：[topics/{topic['name']}.md](topics/{topic['name']}.md)", ""]
        for source in sorted(registry.values(), key=lambda item: item.meta["id"]):
            if source.meta.get("topic") != topic["name"]:
                continue
            target = source.path.relative_to(root / "sources").as_posix()
            quote = counts[source.meta["id"]]
            lines.append(f"- `{source.meta['id']}` — [{source.meta['title']}]({target})（被 {quote} 个页面引用）")
        lines.append("")
    return "\n".join(lines) + "\n"
