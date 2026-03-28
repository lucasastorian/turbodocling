# Corpus

Local benchmark corpus for PDF parsing regression and performance work.

Goals:
- Cover a wider range of document classes than the current golden set.
- Keep source/provenance explicit and reproducible.
- Separate tracked metadata from untracked copyrighted PDFs and generated baselines.

Layout:
- `seed_manifest.json`: tracked seed list with provenance and tags.
- `manifest.lock.json`: generated local inventory with resolved filenames, hashes, and byte sizes.
- `pdfs/`: untracked normalized PDF files.
- `baselines/`: untracked parse-output baselines.
- `benchmarks/`: untracked timing runs and summaries.

Normalization rules:
- Each corpus document gets a stable `id`.
- Files are written to `pdfs/<id>.pdf`.
- Seed entries are either `local_copy` from an existing local file or `download`.
- The lock file records `sha256`, `bytes`, and the resolved local path.

Suggested workflow:
1. Update `seed_manifest.json`.
2. Run `python scripts/corpus_seed.py`.
3. Generate parse baselines for all pages in the corpus.
4. Run large benchmark passes before and after optimizations.

Commands:
- `source .venv/bin/activate && python scripts/corpus_seed.py`
- `source .venv/bin/activate && python scripts/corpus_benchmark.py --update-baselines --runs 1`
- `source .venv/bin/activate && python scripts/corpus_benchmark.py --check-baselines --runs 1`

Notes:
- `scripts/corpus_seed.py --origin-type download` now preserves previously locked local-copy entries instead of truncating `manifest.lock.json`.
- `TD_PY_PROFILE=1` re-enables the Python `_to_segmented_page` timing print if you want that instrumentation during focused profiling.
