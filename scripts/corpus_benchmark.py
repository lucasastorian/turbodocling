#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument


def parse_csv(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def parse_page_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"invalid page range: {part}")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def parse_page(doc: PdfDocument, page_no: int) -> tuple[object, float]:
    if page_no in doc._pages:
        del doc._pages[page_no]

    start = time.perf_counter()
    seg = doc.get_page(page_no, create_words=True, create_textlines=True)
    elapsed = time.perf_counter() - start
    return seg, elapsed


def seg_to_snapshot(seg) -> dict:
    snapshot = {
        "char_count": len(seg.char_cells),
        "word_count": len(seg.word_cells),
        "line_count": len(seg.textline_cells),
        "words": [],
    }
    for word in seg.word_cells:
        snapshot["words"].append(
            {
                "text": word.text,
                "x0": round(word.rect.r_x0, 2),
                "y0": round(word.rect.r_y0, 2),
                "x1": round(word.rect.r_x2, 2),
                "y1": round(word.rect.r_y2, 2),
            }
        )
    return snapshot


def snapshot_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def check_baseline(doc_id: str, results: dict, baseline_dir: Path) -> tuple[bool, list[str]]:
    baseline_path = baseline_dir / f"{doc_id}.json"
    if not baseline_path.exists():
        return True, [f"missing baseline: {baseline_path}"]

    baseline = json.loads(baseline_path.read_text())
    failures: list[str] = []

    for page_no, snapshot in results.items():
        base = baseline.get(page_no)
        if base is None:
            failures.append(f"page {page_no}: missing from baseline")
            continue

        if snapshot["char_count"] != base["char_count"]:
            failures.append(
                f"page {page_no}: char_count {base['char_count']} -> {snapshot['char_count']}"
            )
        if snapshot["word_count"] != base["word_count"]:
            failures.append(
                f"page {page_no}: word_count {base['word_count']} -> {snapshot['word_count']}"
            )
        if snapshot["line_count"] != base["line_count"]:
            failures.append(
                f"page {page_no}: line_count {base['line_count']} -> {snapshot['line_count']}"
            )

        base_words = [word["text"] for word in base["words"]]
        snap_words = [word["text"] for word in snapshot["words"]]
        if base_words != snap_words:
            first_diff = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(base_words, snap_words))
                    if left != right
                ),
                None,
            )
            if first_diff is not None:
                failures.append(
                    f"page {page_no}: word[{first_diff}] "
                    f"{base_words[first_diff]!r} -> {snap_words[first_diff]!r}"
                )
            else:
                failures.append(
                    f"page {page_no}: word_count {len(base_words)} -> {len(snap_words)}"
                )

    return not failures, failures


def select_documents(documents: list[dict], doc_ids: set[str], categories: set[str]) -> list[dict]:
    selected: list[dict] = []
    for document in documents:
        if doc_ids and document["id"] not in doc_ids:
            continue
        if categories and not categories.intersection(document.get("categories", [])):
            continue
        selected.append(document)
    return selected


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def benchmark_document(
    repo_root: Path,
    document: dict,
    runs: int,
    page_filter: list[int] | None,
) -> tuple[dict, dict]:
    pdf_path = (repo_root / document["resolved_path"]).resolve()
    parser = DoclingPdfParser(loglevel="fatal")
    doc = parser.load(path_or_stream=str(pdf_path))
    total_pages = doc.number_of_pages()

    if page_filter is None:
        page_numbers = list(range(1, total_pages + 1))
    else:
        page_numbers = [page for page in page_filter if 1 <= page <= total_pages]

    results: dict[str, dict] = {}
    page_metrics: list[dict] = []

    for page_no in page_numbers:
        best = None
        seg = None
        for _ in range(runs):
            seg, elapsed = parse_page(doc, page_no)
            if best is None or elapsed < best:
                best = elapsed
        assert best is not None
        snapshot = seg_to_snapshot(seg)
        results[str(page_no)] = snapshot
        page_metrics.append(
            {
                "page": page_no,
                "time_s": round(best, 6),
                "char_count": snapshot["char_count"],
                "word_count": snapshot["word_count"],
                "line_count": snapshot["line_count"],
                "hash": snapshot_hash(snapshot),
            }
        )

    total_best_time = round(sum(page["time_s"] for page in page_metrics), 6)
    slowest_page = max(page_metrics, key=lambda item: item["time_s"]) if page_metrics else None
    metrics = {
        "id": document["id"],
        "title": document.get("title"),
        "categories": document.get("categories", []),
        "bytes": document.get("bytes"),
        "sha256": document.get("sha256"),
        "resolved_path": document.get("resolved_path"),
        "page_count": total_pages,
        "benchmarked_pages": len(page_metrics),
        "runs": runs,
        "total_best_time_s": total_best_time,
        "avg_page_time_s": round(total_best_time / len(page_metrics), 6) if page_metrics else 0.0,
        "slowest_page": slowest_page,
        "pages": page_metrics,
    }
    return results, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local PDF corpus.")
    parser.add_argument("--manifest", default="corpus/manifest.lock.json")
    parser.add_argument("--baseline-dir", default="corpus/baselines")
    parser.add_argument("--benchmark-dir", default="corpus/benchmarks")
    parser.add_argument("--doc", action="append", help="Document id filter; repeat or comma-separate")
    parser.add_argument("--category", action="append", help="Category filter; repeat or comma-separate")
    parser.add_argument("--pages", help="Page list or ranges, e.g. 1,2,5-8")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--update-baselines", action="store_true")
    parser.add_argument("--check-baselines", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = (repo_root / args.manifest).resolve()
    baseline_dir = (repo_root / args.baseline_dir).resolve()
    benchmark_dir = (repo_root / args.benchmark_dir).resolve()

    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    documents = manifest.get("documents", [])
    doc_ids = set(parse_csv(args.doc))
    categories = set(parse_csv(args.category))
    page_filter = parse_page_list(args.pages)
    selected = select_documents(documents, doc_ids, categories)

    if not selected:
        print("no documents selected", file=sys.stderr)
        return 1

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_ok = True
    report_documents: list[dict] = []
    all_slow_pages: list[dict] = []

    print(
        f"benchmarking {len(selected)} document(s)"
        f" from {manifest_path.relative_to(repo_root)}"
    )

    for document in selected:
        results, metrics = benchmark_document(
            repo_root=repo_root,
            document=document,
            runs=args.runs,
            page_filter=page_filter,
        )
        report_documents.append(metrics)
        for page in metrics["pages"]:
            all_slow_pages.append(
                {
                    "doc_id": document["id"],
                    "title": document.get("title"),
                    **page,
                }
            )

        slowest = metrics["slowest_page"]
        slowest_label = "n/a"
        if slowest is not None:
            slowest_label = f"{slowest['page']}@{slowest['time_s']:.3f}s"

        print(
            f"{document['id']:<28} pages={metrics['benchmarked_pages']:>3d}"
            f" total={metrics['total_best_time_s']:>8.3f}s"
            f" avg={metrics['avg_page_time_s']:>7.3f}s"
            f" slowest={slowest_label}"
        )

        if args.update_baselines:
            save_json(baseline_dir / f"{document['id']}.json", results)

        if args.check_baselines:
            ok, failures = check_baseline(document["id"], results, baseline_dir)
            if ok:
                print(f"  baseline: PASS")
            else:
                print(f"  baseline: FAIL")
                for failure in failures[:10]:
                    print(f"    {failure}")
                if len(failures) > 10:
                    print(f"    ... {len(failures) - 10} more")
                all_ok = False

    all_slow_pages.sort(key=lambda item: item["time_s"], reverse=True)
    summary = {
        "docs": len(report_documents),
        "pages": sum(document["benchmarked_pages"] for document in report_documents),
        "total_best_time_s": round(
            sum(document["total_best_time_s"] for document in report_documents), 6
        ),
        "slowest_pages": all_slow_pages[:25],
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path.relative_to(repo_root)),
        "runs": args.runs,
        "documents": report_documents,
        "summary": summary,
    }

    save_json(benchmark_dir / f"benchmark-{timestamp}.json", report)
    save_json(benchmark_dir / "latest.json", report)

    print(
        f"summary docs={summary['docs']} pages={summary['pages']}"
        f" total={summary['total_best_time_s']:.3f}s"
    )
    if summary["slowest_pages"]:
        slowest = summary["slowest_pages"][:5]
        print("top slow pages:")
        for page in slowest:
            print(
                f"  {page['doc_id']} page {page['page']}: "
                f"{page['time_s']:.3f}s hash={page['hash']}"
            )

    if args.check_baselines and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
