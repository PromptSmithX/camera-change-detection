"""Validate a canonical benchmark dataset.

Usage:
    python tools/validate_dataset.py --manifest data/benchmark/v1/manifest.json --mode draft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import write_json
from change_detection.dataset.validation import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--mode", choices=("draft", "official"), default="draft")
    parser.add_argument("--strict-hashes", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    report = validate_manifest(
        manifest,
        repo_root=args.root.resolve(),
        mode=args.mode,
        strict_hashes=args.strict_hashes,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if args.report:
        write_json(args.report if args.report.is_absolute() else args.root / args.report, report.to_dict())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
