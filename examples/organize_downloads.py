"""Sort the Downloads folder: an LLM proposes folders, Jev assigns, this script moves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import hunch

SKIP_NAMES = {".DS_Store", ".localized", "$RECYCLE.BIN"}
NO_PILES = {"review", "other", "misc", "unsorted"}


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def list_downloads(root: Path) -> list[dict[str, object]]:
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
            }
        )
    return sorted(rows, key=lambda row: str(row["modified"]), reverse=True)


def unique_dest(folder: Path, name: str) -> Path:
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    index = 1
    while (folder / f"{stem} ({index}){suffix}").exists():
        index += 1
    return folder / f"{stem} ({index}){suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort Downloads into Jev-chosen folders.")
    parser.add_argument("--root", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--limit", type=int, default=0, help="Newest items only. 0 = all.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-taxonomy", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "hunch" / "downloads")
    parser.add_argument("--env", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    args = parser.parse_args()
    load_env(args.env)

    items = list_downloads(args.root)
    if not items:
        raise SystemExit(f"No visible items in {args.root}")
    print(f"Listed {len(items)} top-level items in {args.root}", flush=True)

    jev = hunch.configure(llm=hunch.openrouter(), cache=args.cache, max_workers=args.workers)

    taxonomy_file = args.cache / "taxonomy.json"
    taxonomy = None if args.refresh_taxonomy else _read_taxonomy(taxonomy_file)
    if taxonomy is None:
        print("Asking the LLM for a taxonomy…", flush=True)
        taxonomy = hunch.generate(
            str,
            n=12,
            instructions=(
                "Propose kebab-case folder names that cover this Downloads pile. Group by meaning "
                "(work docs, personal paperwork, books, design assets, installers, archives, media, "
                "project folders), not by extension alone. Include junk for disposable leftovers. "
                "Never review, other, misc, or unsorted: every item must have a real home."
            ),
            context={"listing": [f"{r['kind']}\t{r['ext']}\t{r['name']}" for r in items[:200]]},
        )
        taxonomy = [label for label in dict.fromkeys(taxonomy) if label not in NO_PILES]
        if "junk" not in taxonomy:
            taxonomy.append("junk")
        args.cache.mkdir(parents=True, exist_ok=True)
        taxonomy_file.write_text(json.dumps(taxonomy, indent=2) + "\n")
    print("Taxonomy:", ", ".join(taxonomy), flush=True)

    movable = [r for r in items if not (r["kind"] == "dir" and r["name"] in taxonomy)]
    if args.limit > 0:
        movable = movable[: args.limit]
    if not movable:
        print("Nothing new to file. Destination folders were left alone.")
        return

    print(f"Classifying {len(movable)} items with Jev…", flush=True)
    folders = hunch.classify(
        movable,
        taxonomy,
        instructions=(
            "Which folder should this Downloads item go in? Pick the best fit even if the name is thin. "
            "Camera-roll stills (IMG_, DSC_, screenshots) go with photos or screenshots. Disk images and "
            "installers go in installers. Named project folders and their zips go with projects. "
            "Use junk only for obvious leftovers."
        ),
        detail=True,
    )

    moved = skipped = 0
    counts: dict[str, int] = {}
    for row, answer in zip(movable, folders):
        folder = answer.on(sure=answer.label, split=answer.label, unsure=None)
        if folder is None:
            skipped += 1
            print(f"?  {row['name']}  (unsure: {answer.top2})", flush=True)
            continue
        counts[folder] = counts.get(folder, 0) + 1
        source = args.root / str(row["name"])
        dest = unique_dest(args.root / folder, source.name)
        print(f"{source.name} -> {folder}/{dest.name}", flush=True)
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
        moved += 1

    print()
    for folder, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{count:4d}  {folder}")
    used = jev.usage
    print(f"\n{'Would move' if args.dry_run else 'Moved'} {moved} items. Left {skipped} for review.")
    print(f"Jev {used.model or 'jev'}: {used.calls} calls, {used.hits} cache hits, {used.input_tokens} in / {used.output_tokens} out tokens.")


def _read_taxonomy(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    labels = json.loads(path.read_text())
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        return None
    return labels


if __name__ == "__main__":
    main()
