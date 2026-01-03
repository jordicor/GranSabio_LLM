#!/usr/bin/env python3
"""
Update Cloudflare IP ranges in trusted_proxies.txt

Fetches the latest IP ranges from Cloudflare's official endpoints and updates
the trusted_proxies.txt file while preserving localhost and private network entries.

Usage:
    python tools/update_cloudflare_ips.py [--dry-run] [--output PATH]

Options:
    --dry-run       Show what would be written without modifying the file
    --output PATH   Write to a custom path instead of core/trusted_proxies.txt
"""

import argparse
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Official Cloudflare IP range endpoints
CLOUDFLARE_IPV4_URL = "https://www.cloudflare.com/ips-v4"
CLOUDFLARE_IPV6_URL = "https://www.cloudflare.com/ips-v6"

DEFAULT_OUTPUT = Path(__file__).parent.parent / "core" / "trusted_proxies.txt"

# Static entries to always include
HEADER_TEMPLATE = """# Trusted proxy CIDRs (one per line)
# Source: https://www.cloudflare.com/ips/
# Last updated: {date}
#
# For Cloudflare Tunnel (Zero Trust): traffic arrives from cloudflared daemon,
# typically 127.0.0.1 if running locally. The public Cloudflare IPs below are
# for traditional proxy mode or hybrid setups.

# Localhost (for local dev and Cloudflare Tunnel)
127.0.0.1/32
::1/128

# Private networks (if cloudflared runs on a different internal server)
# Uncomment if needed:
# 10.0.0.0/8
# 172.16.0.0/12
# 192.168.0.0/16
"""


def fetch_ip_ranges(url: str) -> list[str]:
    """Fetch IP ranges from a Cloudflare endpoint."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GranSabio-LLM/1.0 (Cloudflare IP Updater)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8")
            return [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return []


def generate_trusted_proxies_content(ipv4_ranges: list[str], ipv6_ranges: list[str]) -> str:
    """Generate the complete trusted_proxies.txt content."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = [HEADER_TEMPLATE.format(date=date_str)]

    # Add IPv4 ranges
    lines.append("# Cloudflare IPv4 ranges")
    for cidr in sorted(ipv4_ranges):
        lines.append(cidr)

    lines.append("")

    # Add IPv6 ranges
    lines.append("# Cloudflare IPv6 ranges")
    for cidr in sorted(ipv6_ranges):
        lines.append(cidr)

    lines.append("")  # Trailing newline

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Update Cloudflare IP ranges in trusted_proxies.txt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without modifying the file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    print("Fetching Cloudflare IPv4 ranges...")
    ipv4_ranges = fetch_ip_ranges(CLOUDFLARE_IPV4_URL)
    if not ipv4_ranges:
        print("Failed to fetch IPv4 ranges. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(ipv4_ranges)} IPv4 ranges")

    print("Fetching Cloudflare IPv6 ranges...")
    ipv6_ranges = fetch_ip_ranges(CLOUDFLARE_IPV6_URL)
    if not ipv6_ranges:
        print("Failed to fetch IPv6 ranges. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(ipv6_ranges)} IPv6 ranges")

    content = generate_trusted_proxies_content(ipv4_ranges, ipv6_ranges)

    if args.dry_run:
        print("\n--- DRY RUN: Would write the following content ---\n")
        print(content)
        print("--- END DRY RUN ---")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"\nUpdated {args.output}")
        print(f"  IPv4 ranges: {len(ipv4_ranges)}")
        print(f"  IPv6 ranges: {len(ipv6_ranges)}")
        print(f"  Total CIDRs: {len(ipv4_ranges) + len(ipv6_ranges) + 2} (including localhost)")


if __name__ == "__main__":
    main()
