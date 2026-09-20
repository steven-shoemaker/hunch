"""Sort the Downloads folder: an LLM proposes folders, Jev assigns, this script moves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hunch import ask, connect, draft, openrouter, over, role

SKIP_NAMES = {".DS_Store", ".localized", "$RECYCLE.BIN"}


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def list_downloads(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in root.iterdir():
        if path.name.startswith(".") or path.name in SKIP_NAMES:
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "name": path.name,
                "kind": "dir" if path.is_dir() else "file",
                "ext": "" if path.is_dir() else path.suffix.lower(),
                "bytes": info.st_size,
                "modified": datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat(),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows).sort_values("modified", ascending=False).reset_index(drop=True)


def unique_dest(folder: Path, name: str) -> Path:
    dest = folder / name
    if not dest.exists():
        return dest
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 1
    while True:
        candidate = folder / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_item(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort Downloads into Jev-chosen folders.")
    parser.add_argument("--root", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--limit", type=int, default=0, help="Newest items only. 0 = all.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-taxonomy", action="store_true")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "hunch" / "downloads",
    )
    parser.add_argument("--env", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    args = parser.parse_args()

    load_env(args.env)
    items = list_downloads(args.root)
    if items.empty:
        raise SystemExit(f"No visible items in {args.root}")

    listing = [
        f"{row.kind}\t{row.ext}\t{row.name}"
        for row in items.head(200).itertuples(index=False)
    ]
    jev = connect(llm=openrouter(model="z-ai/glm-5.3-flash"), cache=args.cache)
    taxonomist = role(
        "Propose 8–16 kebab-case folder names that cover this Downloads pile. "
        "Group by meaning (work docs, personal paperwork, books, design assets, "
        "installers, archives, media, project folders), not by file extension alone. "
        "Include junk for disposable leftovers. Do not include review, other, or misc — "
        "every item must have a real home. No duplicates.",
        emit=list[str],
    )

    print(f"Listed {len(items)} top-level items in {args.root}", flush=True)
    saved = None if args.refresh_taxonomy else _read_taxonomy(args.cache)
    if saved:
        taxonomy = saved
        print("Using saved taxonomy:", ", ".join(taxonomy), flush=True)
    else:
        print("Asking OpenRouter for a taxonomy…", flush=True)
        with jev.session():
            taxonomy = draft(listing, taxonomist).labels
        taxonomy = [label for label in taxonomy if label not in {"review", "other", "misc", "unsorted"}]
        if "junk" not in taxonomy:
            taxonomy.append("junk")
        _write_taxonomy(args.cache, taxonomy)
        print("Taxonomy:", ", ".join(taxonomy), flush=True)

    dest_names = set(taxonomy)
    movable = items[~items["name"].isin(dest_names) | (items["kind"] != "dir")].copy()
    sample = movable if args.limit <= 0 else movable.head(args.limit)
    if sample.empty:
        print("Nothing new to file. Destination folders were left alone.")
        print(f"Jev usage: {jev.usage.calls} calls, {jev.usage.hits} cache hits.")
        return
    lookup = {row["name"]: row for row in sample.to_dict(orient="records")}
    print(f"Classifying {len(sample)} new items with Jev ({args.workers} workers)…", flush=True)

    @over(sample, "name")
    def classify(name):
        row = lookup[str(name)]
        state = {
            "name": row["name"],
            "kind": row["kind"],
            "ext": row["ext"],
            "bytes": row["bytes"],
            "modified": row["modified"],
        }
        folder = ask(
            state,
            "Which folder should this Downloads item go in?",
            among=taxonomy,
            by=(
                "Pick the best fitting folder even if the name is thin. "
                "Camera-roll stills (IMG_, DSC_, screenshots) go in a photos or screenshots folder. "
                "App disk images and installers go in installers. "
                "Named project folders and their zips go in side-projects or dev-projects. "
                "Use junk only for obvious leftovers or disposable exports. "
                "Never pick a review, other, or unsorted pile."
            ),
        )
        return {
            "folder": folder.top,
            "shape": folder.shape,
            "confidence": round(folder.confidence, 3),
            "p": round(folder.p, 3),
        }

    results = classify.run(
        jev,
        max_workers=args.workers,
        on_item=lambda done, total, _value: print(f"  classified {done}/{total}", flush=True)
        if done == total or done % 25 == 0
        else None,
    )

    moved = 0
    skipped = 0
    for row in results.itertuples(index=False):
        folder = getattr(row, "folder", None)
        if not isinstance(folder, str) or not folder:
            skipped += 1
            continue
        source = Path(row.path)
        if not source.exists():
            skipped += 1
            continue
        dest_dir = args.root / folder
        if source.resolve() == dest_dir.resolve():
            skipped += 1
            continue
        dest = unique_dest(dest_dir, source.name)
        print(f"{source.name} -> {folder}/{dest.name}", flush=True)
        if not args.dry_run:
            move_item(source, dest)
        moved += 1

    print()
    print(results.groupby("folder").size().sort_values(ascending=False).to_string())
    print()
    action = "Would move" if args.dry_run else "Moved"
    used = jev.usage
    print(f"{action} {moved} items. Skipped {skipped}.")
    print(
        f"Jev {used.model or 'jev'}: {used.calls} calls, {used.hits} cache hits, "
        f"{used.input_tokens} in / {used.output_tokens} out tokens."
    )


def _read_taxonomy(cache: Path) -> list[str] | None:
    path = cache / "taxonomy.json"
    if not path.is_file():
        return None
    labels = json.loads(path.read_text())
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        return None
    return labels


def _write_taxonomy(cache: Path, labels: list[str]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "taxonomy.json").write_text(json.dumps(labels, indent=2) + "\n")


if __name__ == "__main__":
    main()
