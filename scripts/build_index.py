"""由页面与来源元数据生成索引，正文内容不复制到索引。"""
import argparse
import sys

from wiki_common import ROOT, pages, render_page_index, render_source_index, sources, vocabulary


def render():
    vocab = vocabulary(ROOT)
    records = pages(ROOT)
    return [(ROOT / "index.generated.md", render_page_index(ROOT, records, vocab)),
            (ROOT / "sources/INDEX.generated.md",
             render_source_index(ROOT, sources(ROOT), records, vocab))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in render():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_text(content, encoding="utf-8")
    if args.check:
        if stale:
            print(f"索引缺失或过期: {', '.join(stale)}；请运行 python3 scripts/build_index.py", file=sys.stderr)
            sys.exit(1)
        print("索引与页面元数据一致")
