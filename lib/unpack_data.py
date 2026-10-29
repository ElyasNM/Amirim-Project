"""Unpack the data archives in data/archives/ into DATA_ROOT (default: ./data).

    python lib/unpack_data.py            # everything (about 1.4 GB unpacked)
    python lib/unpack_data.py klar Llama-3.1   # only archives whose name contains one of the words
    python lib/unpack_data.py --list     # show archives and sizes

Archives are extracted relative to DATA_ROOT, so the result matches the layout described in the README.
"""
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_ROOT, REPO_ROOT

ARCHIVES = REPO_ROOT / "data" / "archives"


def safe_members(tf):
    for m in tf.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"unsafe archive member: {m.name}")
        yield m


def main(args):
    archives = sorted(ARCHIVES.glob("*.tar.gz"))
    if "--list" in args:
        for a in archives:
            print(f"{a.stat().st_size / 1e6:8.1f} MB  {a.name}")
        return
    if args:
        archives = [a for a in archives if any(w in a.name for w in args)]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for a in archives:
        with tarfile.open(a) as tf:
            members = list(safe_members(tf))
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(DATA_ROOT, members=members, **kw)
        print(f"unpacked {a.name} ({len(members)} entries) -> {DATA_ROOT}")


if __name__ == "__main__":
    main(sys.argv[1:])
