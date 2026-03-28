#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "turbodocling-corpus-seed/1.0"
        },
    )
    with urllib.request.urlopen(request) as response:
        if response.status != 200:
            raise RuntimeError(f"download failed for {url} with HTTP {response.status}")
        with dest.open("wb") as out:
            shutil.copyfileobj(response, out)


def ensure_pdf(path: Path) -> None:
    header = path.read_bytes()[:5]
    if header != b"%PDF-":
        raise RuntimeError(f"{path} does not look like a PDF")


def materialize_entry(repo_root: Path, entry: dict, pdf_dir: Path, refresh: bool) -> dict:
    entry_id = entry["id"]
    origin = entry["origin"]
    dest = pdf_dir / f"{entry_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if origin["type"] == "local_copy":
        src = (repo_root / origin["path"]).resolve()
        if not src.exists():
            raise FileNotFoundError(f"missing local source: {src}")
        if refresh or not dest.exists():
            shutil.copy2(src, dest)
    elif origin["type"] == "download":
        if refresh or not dest.exists():
            download(origin["url"], dest)
    else:
        raise ValueError(f"unsupported origin type: {origin['type']}")

    ensure_pdf(dest)
    stat = dest.stat()
    locked = dict(entry)
    locked["resolved_path"] = str(dest.relative_to(repo_root))
    locked["bytes"] = stat.st_size
    locked["sha256"] = sha256_file(dest)
    return locked


def load_existing_documents(lock_path: Path) -> dict[str, dict]:
    if not lock_path.exists():
        return {}
    try:
        lock = json.loads(lock_path.read_text())
    except json.JSONDecodeError:
        return {}
    documents = lock.get("documents", [])
    return {doc["id"]: doc for doc in documents if "id" in doc}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the local PDF benchmark corpus.")
    parser.add_argument(
        "--manifest",
        default="corpus/seed_manifest.json",
        help="Path to the seed manifest JSON",
    )
    parser.add_argument(
        "--pdf-dir",
        default="corpus/pdfs",
        help="Directory for normalized local PDFs",
    )
    parser.add_argument(
        "--lock",
        default="corpus/manifest.lock.json",
        help="Generated lock file path",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-copy and re-download all entries even if already present",
    )
    parser.add_argument(
        "--origin-type",
        choices=["all", "local_copy", "download"],
        default="all",
        help="Restrict processing to one origin type",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = (repo_root / args.manifest).resolve()
    pdf_dir = (repo_root / args.pdf_dir).resolve()
    lock_path = (repo_root / args.lock).resolve()

    manifest = json.loads(manifest_path.read_text())
    documents = manifest.get("documents", [])

    existing_docs = load_existing_documents(lock_path)
    locked_docs = []
    failures = []
    for entry in documents:
        if args.origin_type != "all" and entry["origin"]["type"] != args.origin_type:
            existing = existing_docs.get(entry["id"])
            if existing is not None:
                locked_docs.append(existing)
            continue
        try:
            locked_docs.append(
                materialize_entry(repo_root, entry, pdf_dir, refresh=args.refresh)
            )
            print(f"seeded {entry['id']}")
        except Exception as exc:
            failures.append({"id": entry.get("id"), "error": str(exc)})
            print(f"failed {entry.get('id')}: {exc}", file=sys.stderr)

    lock = {
        "version": manifest.get("version", 1),
        "documents": locked_docs,
        "failures": failures,
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    if failures:
        print(f"completed with {len(failures)} failure(s)", file=sys.stderr)
        return 1

    print(f"wrote {lock_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
