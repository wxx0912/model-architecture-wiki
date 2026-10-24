"""检查页面契约、来源页契约、引用关系与本地链接。"""
import argparse
import sys
import urllib.request

from wiki_common import ROOT, sources, validate


def unreachable(proxy=None, timeout=15):
    """只检查来源 URL 可达性；可达不等于内容已复核，不可达也不等于来源无效。"""
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    errors = []
    for source_id, source in sorted(sources(ROOT).items()):
        request = urllib.request.Request(source.meta["url"], method="HEAD",
                                         headers={"User-Agent": "quantization-wiki-validate"})
        try:
            opener.open(request, timeout=timeout)
        except Exception as exc:
            errors.append(f"URL {source_id}: 不可达 {exc}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-external", action="store_true",
                        help="额外检查来源 URL 可达性（需要网络与 http/https 代理）")
    parser.add_argument("--proxy", help="http/https 代理地址；socks5 代理不被标准库支持")
    args = parser.parse_args()
    errors, warnings, page_count, source_count = validate()
    if args.check_external:
        errors += unreachable(args.proxy)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{page_count} pages, {source_count} sources, {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(bool(errors))
