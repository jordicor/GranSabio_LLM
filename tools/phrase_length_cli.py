#!/usr/bin/env python
"""CLI tool to find optimal phrase length for smart-edit uniqueness."""
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_edit.locators import find_optimal_phrase_length


def main():
    parser = argparse.ArgumentParser(
        description="Find minimum n-gram length where ALL n-grams in text are unique."
    )
    parser.add_argument("file", help="Path to text file to analyze")
    parser.add_argument(
        "--min-n", type=int, default=4, help="Minimum n-gram size (default: 4)"
    )
    parser.add_argument(
        "--max-n", type=int, default=64, help="Maximum n-gram size (default: 64)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show additional info"
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    text = file_path.read_text(encoding="utf-8")

    if args.verbose:
        word_count = len(text.split())
        print(f"File: {args.file}", file=sys.stderr)
        print(f"Words: {word_count}", file=sys.stderr)
        print(f"Range: {args.min_n}-{args.max_n}", file=sys.stderr)

    result = find_optimal_phrase_length(text, min_n=args.min_n, max_n=args.max_n)

    if result is None:
        if args.verbose:
            print(f"No unique length found in range {args.min_n}-{args.max_n}", file=sys.stderr)
        print("None")
        sys.exit(2)
    else:
        print(result)


if __name__ == "__main__":
    main()
