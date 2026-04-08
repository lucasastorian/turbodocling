"""
CLI entry point for Turbodocling.

Usage:
    python -m turbodocling my_document.pdf -o output/
    python -m turbodocling my_document.pdf --device cpu --workers 4
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="turbodocling",
        description="Process a PDF document locally using the Turbodocling pipeline.",
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device (default: auto-detect)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of preprocessing workers (default: CPU count)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Pages per preprocessing batch (default: auto)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found", file=sys.stderr)
        sys.exit(1)

    from turbodocling.local_runner import run_local

    result = run_local(
        pdf_path=args.pdf,
        output_dir=args.output,
        device=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    print(f"\nDone. {result.total_pages} pages in {result.wall_time_s:.1f}s on {result.device}.")
    print(f"  {result.md_path}")
    print(f"  {result.elements_path}")


if __name__ == "__main__":
    main()
