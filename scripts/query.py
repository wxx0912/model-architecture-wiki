"""按名称、别名、机制标签或问题关键词检索知识页。"""
import argparse
import json

from wiki_common import ROOT, pages, search, vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--tag")
    parser.add_argument("--type", dest="kind")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于零")
    vocab = vocabulary(ROOT)
    kinds = [item["name"] for item in vocab["page_types"]]
    if args.kind and args.kind not in kinds:
        parser.error(f"--type 取值须属 {kinds}")
    tag = None
    if args.tag:
        tag = {item.casefold(): item for item in vocab["tags"]}.get(args.tag.casefold())
        if tag is None:
            parser.error(f"--tag 未登记 {args.tag}；标签词表见 data/vocabulary.yaml")
    if not args.query and not tag and not args.kind:
        print("请给出关键词，或用 --tag <标签> / --type <页面类型> 筛选；可用项见 index.generated.md。")
        return
    rows = [{"id": page.meta["id"], "title": page.meta["title"],
             "path": page.path.relative_to(ROOT).as_posix(), "type": page.meta["type"],
             "tags": page.meta["tags"], "sources": page.meta["sources"],
             "score": score, "weak": weak}
            for score, page, weak in search(pages(ROOT), args.query, tag, args.kind, args.limit)]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        print(f"未命中 {args.query!r}。")
        print("关键词只匹配页面名称、别名（aliases）和标签；没有命中可能是关键词写法不同，也可能是本库没有该知识。")
        print(f"可用标签：{'、'.join(vocab['tags'])}")
        print("分类入口见 index.generated.md。确认没有相关知识时按“未知”上报，不要编造。")
        return
    if rows[0]["weak"]:
        print("# 名称/标签未完全命中，以下结果含正文匹配，属弱相关；建议改用准确名称或 --tag")
    for page_type in vocab["page_types"]:
        group = [row for row in rows if row["type"] == page_type["name"]]
        if not group:
            continue
        print(f"## {page_type['label']}")
        for row in group:
            print(f"{row['id']}\t{row['title']}\t{row['path']}")
    print("用 python3 scripts/get_page.py <id> 读取页面，加 --follow-sources 展开来源。")


if __name__ == "__main__":
    main()
