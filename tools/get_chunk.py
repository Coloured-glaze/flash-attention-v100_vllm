"""
Download & merge chunked wheel files from a GitHub release.

The build workflow splits .whl files larger than 1 GiB into chunks named
`<name>.whl.part000`, `<name>.whl.part001`, ... along with a
`<name>.whl.sha256` sidecar (SHA-256 of the *original* wheel).

This script:
  1. Lists release assets via the GitHub API.
  2. Downloads `.partNNN` chunks in order and concatenates them.
  3. Verifies the SHA-256 against the `.sha256` sidecar.
  4. Optionally `pip install`s the resulting wheel.

Standard library only (Python 3.7+).

Usage:
    # By release URL (public repo):
    python get_chunk.py https://github.com/OWNER/REPO/releases/tag/TAG

    # Or by repo + tag:
    python get_chunk.py --repo OWNER/REPO --tag TAG

    # Download, merge, then pip install in one shot:
    python get_chunk.py <release_url> --install

    # Download, merge, install, then remove the .whl file:
    python get_chunk.py <release_url> --install --cleanup

    # Custom output directory:
    python get_chunk.py <release_url> -o ./downloads

    # Private repo / higher rate limit:
    python get_chunk.py <release_url> --token ghp_xxx

Examples:
    python get_chunk.py https://github.com/me/torch-builds/releases/tag/torch-v2.13.0-cu129-2026-08-03
    python get_chunk.py --repo me/torch-builds --tag torch-v2.13.0-cu129-2026-08-03 --install
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# `<name>.whl.part000`, `<name>.whl.part001`, ...
CHUNK_RE = re.compile(r"^(.+)\.part(\d+)$")


def parse_release_url(url: str):
    """https://github.com/OWNER/REPO/releases/tag/TAG -> (OWNER/REPO, TAG)."""
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/releases/tag/(.+)",
        url.rstrip("/"),
    )
    if not m:
        raise ValueError(
            f"Unrecognized release URL: {url}\n"
            f"Expected: https://github.com/OWNER/REPO/releases/tag/TAG"
        )
    return f"{m.group(1)}/{m.group(2)}", m.group(3)


def gh_get(url: str, token=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "get_chunk.py",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"GitHub API {e.code} for {url}:\n{body}") from None


def download(url: str, dest: Path, token=None, label=None):
    """Stream a URL to disk with a 1 MiB progress bar."""
    headers = {"User-Agent": "get_chunk.py"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    label = label or dest.name
    with urlopen(Request(url, headers=headers)) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        while True:
            buf = resp.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                sys.stdout.write(
                    f"\r  {label}: {got/1e6:.1f}/{total/1e6:.1f} MB ({got*100/total:.1f}%)"
                )
            else:
                sys.stdout.write(f"\r  {label}: {got/1e6:.1f} MB")
            sys.stdout.flush()
    sys.stdout.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def group_assets(assets):
    """Split assets into chunked groups, plain files, and sha256 sidecars."""
    chunked, plain, sha = {}, [], {}
    for a in assets:
        name = a["name"]
        m = CHUNK_RE.match(name)
        if m:
            chunked.setdefault(m.group(1), []).append((int(m.group(2)), a))
        elif name.endswith(".sha256"):
            sha[name[:-7]] = a
        else:
            plain.append(a)
    for base in chunked:
        chunked[base].sort(key=lambda x: x[0])
    return chunked, plain, sha


def fetch_sha(asset: dict, out_dir: Path, token=None) -> str:
    tmp = out_dir / f".{asset['name']}.tmp"
    download(asset["browser_download_url"], tmp, token=token, label=asset["name"])
    digest = tmp.read_text().strip().split()[0]
    tmp.unlink()
    return digest


def merge_chunks(base: str, parts, out_dir: Path, token=None) -> Path:
    out = out_dir / base
    print(f"\n[MERGE] {len(parts)} chunks -> {out}")
    if out.exists():
        print(f"  Overwriting existing {out.name}")
    total = 0
    with open(out, "wb") as wf:
        for idx, asset in parts:
            name = asset["name"]
            print(f"  chunk {name} ({asset['size']} bytes)")
            tmp = out_dir / f".{name}.dl"
            download(asset["browser_download_url"], tmp, token=token, label=name)
            with open(tmp, "rb") as cf:
                while True:
                    buf = cf.read(1024 * 1024)
                    if not buf:
                        break
                    wf.write(buf)
                    total += len(buf)
            tmp.unlink()
    print(f"  -> {out.name} ({total} bytes)")
    return out


def verify(out: Path, expected: str) -> bool:
    actual = sha256_file(out)
    if expected == actual:
        print(f"  sha256 OK: {actual}")
        return True
    print(f"  sha256 MISMATCH!\n    expected: {expected}\n    actual:   {actual}")
    return False


def main():
    ap = argparse.ArgumentParser(
        description="Download & merge chunked wheel files from a GitHub release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("release_url", nargs="?", help="GitHub release URL")
    ap.add_argument("--repo", help="owner/repo (alternative to release_url)")
    ap.add_argument("--tag", help="release tag (alternative to release_url)")
    ap.add_argument("-o", "--output-dir", default=".", help="output directory (default: .)")
    ap.add_argument(
        "--install",
        action="store_true",
        help="pip install the merged .whl file(s) after download",
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the merged .whl file(s) after successful install (use with --install)",
    )
    ap.add_argument("--token", help="GitHub token (private repos / higher rate limit)")
    args = ap.parse_args()

    if args.release_url:
        repo, tag = parse_release_url(args.release_url)
    elif args.repo and args.tag:
        repo, tag = args.repo, args.tag
    else:
        ap.error("Provide a release_url, OR both --repo and --tag")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {repo}")
    print(f"Tag:  {tag}")
    print(f"Out:  {out_dir.resolve()}")

    rel = gh_get(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        token=args.token,
    )
    assets = rel.get("assets", [])
    if not assets:
        print("No assets found in this release.")
        return 1
    print(f"\n{len(assets)} asset(s):")
    for a in assets:
        print(f"  {a['name']}  ({a['size']} bytes)")

    chunked, plain, sha = group_assets(assets)
    print(
        f"\nGrouped: {len(chunked)} chunked file(s), "
        f"{len(plain)} plain file(s), {len(sha)} sha256 sidecar(s)"
    )

    whls = []

    # Chunked files: download parts in order and concatenate.
    for base, parts in chunked.items():
        out = merge_chunks(base, parts, out_dir, token=args.token)
        if base in sha:
            expected = fetch_sha(sha[base], out_dir, token=args.token)
            if not verify(out, expected):
                return 1
        else:
            print("  (no .sha256 asset, skipping verification)")
        if out.suffix == ".whl":
            whls.append(out)

    # Plain (non-chunked) assets: download as-is.
    for asset in plain:
        name = asset["name"]
        out = out_dir / name
        print(f"\n[DOWNLOAD] {name} ({asset['size']} bytes) -> {out}")
        download(asset["browser_download_url"], out, token=args.token, label=name)
        if name in sha:
            expected = fetch_sha(sha[name], out_dir, token=args.token)
            if not verify(out, expected):
                return 1
        if name.endswith(".whl"):
            whls.append(out)

    if args.install:
        if not whls:
            print("\nNo .whl files to install.")
        for whl in whls:
            print(f"\n[INSTALL] pip install {whl}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", str(whl)]
            )
            if args.cleanup:
                print(f"  [CLEANUP] removing {whl.name}")
                whl.unlink()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HTTPError, URLError) as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(2)
