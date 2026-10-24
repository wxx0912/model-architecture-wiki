"""按名称、别名、机制标签或问题关键词检索。"""
import argparse
import json

from wiki_common import ROOT, alias_pages, pages, search, vocabulary


def profile_index(root):
    """读取可选的领域事实源 data/architecture-profiles.json；文件不存在时返回 None。"""
    path = root / "data" / "architecture-profiles.json"
    if not path.is_file():
        return None
    profiles = json.loads(path.read_text(encoding="utf-8")).get("profiles", [])
    page_of, features, classifications = {}, {}, {}
    for profile in profiles:
        page = str(profile.get("knowledge_page"))
        page_of[str(profile.get("id"))] = page
        features.setdefault(page, set()).update(str(item) for item in profile.get("features", []))
        classifications.setdefault(page, set()).update(
            str(item.get("classification")) for item in profile.get("evidence", []) if isinstance(item, dict))
    return page_of, features, classifications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--tag")
    parser.add_argument("--type", dest="kind")
    parser.add_argument("--alias", help="问题级别名键，见 data/vocabulary.yaml")
    parser.add_argument("--profile", help="限定到某个结构 profile 的知识页面")
    parser.add_argument("--feature", help="按结构 profile 特征过滤")
    parser.add_argument("--evidence", help="按结构 profile 证据分类过滤")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于零")
    vocab = vocabulary(ROOT)
    kinds = [item["name"] for item in vocab["page_types"]]
    if args.kind and args.kind not in kinds:
        parser.error(f"--type 取值须属 {kinds}")
    ids = None
    if args.alias:
        ids = alias_pages(vocab, args.alias)
        if not ids:
            parser.error(f"--alias 未登记 {args.alias}；可用键 {sorted(vocab['alias_expansions'])}")
    records = pages(ROOT)
    if args.profile or args.feature or args.evidence:
        index = profile_index(ROOT)
        if index is None:
            parser.error("本仓库没有 data/architecture-profiles.json，不支持 --profile/--feature/--evidence")
        page_of, features, classifications = index
        if args.profile:
            if args.profile not in page_of:
                parser.error(f"--profile 未登记 {args.profile}；可用 profile {sorted(page_of)}")
            records = [page for page in records if page.meta["id"] == page_of[args.profile]]
        else:
            if args.feature:
                records = [page for page in records if args.feature in page.meta["tags"]
                           or args.feature in features.get(page.meta["id"], set())]
            if args.evidence:
                records = [page for page in records
                           if args.evidence in classifications.get(page.meta["id"], set())]
    rows = [{"id": page.meta["id"], "title": page.meta["title"],
             "path": page.path.relative_to(ROOT).as_posix(), "type": page.meta["type"],
             "tags": page.meta["tags"], "sources": page.meta["sources"], "score": score}
            for score, page in search(records, args.query, args.tag, args.kind, args.limit, ids)]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        print(f"{row['id']}\t{row['title']}\t{row['path']}")
    if not rows:
        print("未命中；尝试方法别名、较短的问题关键词、--tag 或 --alias。未命中不代表方法不可用。")


if __name__ == "__main__":
    main()
