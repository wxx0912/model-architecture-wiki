"""比对共享文件与对端仓库是否逐字节一致（CONVENTIONS.md 第 8 节）。"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = (
    "CONVENTIONS.md",
    "sources/SPEC.md",
    "scripts/wiki_common.py",
    "scripts/query.py",
    "scripts/get_page.py",
    "scripts/validate.py",
    "scripts/build_index.py",
    "scripts/check_shared.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", required=True, help="另一仓库的本地路径")
    args = parser.parse_args()
    peer = Path(args.peer).resolve()
    problems = []
    for name in SHARED:
        local, remote = ROOT / name, peer / name
        if not local.is_file():
            problems.append(f"{name}: 本仓库缺失")
        elif not remote.is_file():
            problems.append(f"{name}: 对端 {peer} 缺失")
        elif digest(local) != digest(remote):
            problems.append(f"{name}: 内容不一致")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"{len(SHARED)} 个共享文件与 {peer.name} 逐字节一致")


if __name__ == "__main__":
    main()
