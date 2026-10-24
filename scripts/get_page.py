"""按稳定 ID 读取一个页面，可同时展开本地来源页。"""
import argparse

from wiki_common import ROOT, pages, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id")
    parser.add_argument("--follow-sources", action="store_true")
    args = parser.parse_args()
    found = [page for page in pages(ROOT) if page.meta["id"] == args.id]
    if len(found) != 1:
        parser.error(f"页面 ID 应唯一命中，实际 {len(found)} 个: {args.id}")
    page = found[0]
    print(page.path.read_text(encoding="utf-8"))
    if not args.follow_sources:
        return
    registry = sources(ROOT)
    for source_id in page.meta["sources"]:
        source = registry[source_id]
        print(f"\n===== {source_id} · {source.path.relative_to(ROOT).as_posix()} =====")
        print(source.body)
    print("\n来源 URL（人工核对用）：")
    for source_id in page.meta["sources"]:
        print(f"- {source_id}: {registry[source_id].meta['url']}")


if __name__ == "__main__":
    main()
