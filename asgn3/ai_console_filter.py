#!/usr/bin/env python3
"""
Filter stdin by excluding lines that match a pattern (regex by default),
streaming output line-by-line across Linux, macOS, and Windows.

Highlights
----------
- Cross-platform streaming: flushes stdout after every line and enables line buffering.
- Flexible matching: regex (default) or fixed-string, case-insensitive, or keep-only matches.
- Encoding control (useful on Windows/CMD/PowerShell): --encoding and --errors.
- Graceful pipelines: handles downstream closures (e.g., head) without noisy tracebacks.

Windows tips
------------
- Real-time pipelines often depend on the PRODUCER flushing promptly. For Python producers, use `python -u`.
- PowerShell (streaming from a file, like `tail -f`):
  Get-Content -Path .\app.log -Wait | py .\filter_stdin.py ERROR
- CMD examples:
  type app.log | py filter_stdin.py ERROR
  py -u app.py | py filter_stdin.py -i warning
- Encoding: If you see mojibake, try `--encoding utf-8` or set `PYTHONIOENCODING=utf-8`.

Examples
--------
# Exclude lines containing DEBUG (case-sensitive)
myapp | python filter_stdin.py DEBUG

# Exclude lines matching HTTP 4xx or 5xx statuses (regex)
cat access.log | python filter_stdin.py "\\s(4\\d{2}|5\\d{2})\\s"

# Fixed-string match (no regex)
cat data.txt | python filter_stdin.py -F "a.b"

# Case-insensitive filtering
cat app.log | python filter_stdin.py -i error

# Keep only matching lines
cat app.log | python filter_stdin.py --keep-matching ERROR

# Specify encoding (helpful on Windows terminals)
py -u producer.py | py filter_stdin.py --encoding utf-8 PATTERN
"""

import argparse
import os
import re
import signal
import sys
from typing import Iterator, Optional


def iter_stdin_lines() -> Iterator[str]:
    """Yield lines from stdin (including trailing newline if present)."""
    for line in sys.stdin:
        yield line


def configure_streams(encoding: Optional[str], errors: str) -> None:
    """Configure stdin/stdout for cross-platform streaming and encoding stability.

    - Enable line-buffered stdout for immediate line-by-line output.
    - Optionally reconfigure stdin/stdout encoding to avoid mojibake on Windows.
    """
    # Try modern reconfigure API (Python 3.7+), otherwise fall back to fdopen.
    # 1) Encoding settings
    if encoding:
        try:
            sys.stdin.reconfigure(encoding=encoding, errors=errors, newline=None)
        except Exception:
            try:
                sys.stdin = os.fdopen(
                    sys.stdin.fileno(),
                    "r",
                    buffering=-1,
                    encoding=encoding,
                    errors=errors,
                )
            except Exception:
                pass
        try:
            sys.stdout.reconfigure(encoding=encoding, errors=errors)
        except Exception:
            try:
                sys.stdout = os.fdopen(
                    sys.stdout.fileno(),
                    "w",
                    buffering=1,
                    encoding=encoding,
                    errors=errors,
                )
            except Exception:
                pass

    # 2) Ensure line-buffered stdout for streaming behavior
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        try:
            # buffering=1 => line-buffered for text mode
            sys.stdout = os.fdopen(
                sys.stdout.fileno(),
                "w",
                buffering=1,
                encoding=getattr(sys.stdout, "encoding", None) or None,
            )
        except Exception:
            # As a last resort, we'll explicitly flush after each write.
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter stdin by excluding lines that match a pattern (regex by default).",
    )
    parser.add_argument(
        "pattern",
        help="Pattern to match. Regex by default; use -F/--fixed-string for literal matching.",
    )
    parser.add_argument(
        "-F",
        "--fixed-string",
        action="store_true",
        help="Treat the pattern as a literal string (no regex metacharacters).",
    )
    parser.add_argument(
        "-i",
        "--ignore-case",
        action="store_true",
        help="Case-insensitive matching.",
    )
    parser.add_argument(
        "--keep-matching",
        action="store_true",
        help="Keep ONLY lines that match (instead of excluding them).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help=(
            "Re-encode stdin/stdout using the given encoding (e.g., utf-8). Useful on Windows when console"
            " code pages cause mojibake. If omitted, the system defaults are used."
        ),
    )
    parser.add_argument(
        "--errors",
        choices=["strict", "replace", "ignore"],
        default="replace",
        help="Error handling for encoding/decoding (default: replace).",
    )

    args = parser.parse_args()

    # Configure SIGPIPE so downstream consumers closing early (e.g., `head`) don't cause tracebacks.
    # On Windows SIGPIPE may not be available; this is safely wrapped.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    # Streaming and encoding setup
    configure_streams(args.encoding, args.errors)

    flags = re.MULTILINE
    if args.ignore_case:
        flags |= re.IGNORECASE

    pattern = args.pattern
    try:
        if args.fixed_string:
            pattern = re.escape(pattern)
        regex = re.compile(pattern, flags)
    except re.error as e:
        print(f"Invalid pattern: {e}", file=sys.stderr)
        return 2

    try:
        for line in iter_stdin_lines():
            # Match against the line content without the trailing newline
            content = line[:-1] if line.endswith("\n") else line
            matched = regex.search(content) is not None
            keep = matched if args.keep_matching else not matched
            if keep:
                try:
                    sys.stdout.write(line)
                    try:
                        sys.stdout.flush()  # ensure immediate visibility on all platforms
                    except Exception:
                        pass
                except BrokenPipeError:
                    # Downstream pipe closed; exit cleanly.
                    return 0
    except KeyboardInterrupt:
        # Graceful exit on Ctrl-C
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
