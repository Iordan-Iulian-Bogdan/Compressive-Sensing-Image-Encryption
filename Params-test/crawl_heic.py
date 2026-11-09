#!/usr/bin/env python3
"""Crawl pages for HEIC/HEIF photos shot on an iPhone 13 or newer (judged by EXIF).

  python Params-test/crawl_heic.py https://example.com/gallery/ --out heic_crawl --max-pages 200

Needs: pip install pillow pillow-heif
Keeps only files whose EXIF Make is Apple and Model is "iPhone N..." with N >= 13
(or "iPhone Air"). Everything else is dropped after the check. Stays on the seed's
host unless --any-domain, honours robots.txt, waits --delay seconds between requests.
Writes <out>/<sha1>.heic and <out>/manifest.tsv (url, model, taken, has_gps, bytes).
The files are the originals: they can carry GPS and owner data, and they belong to
whoever shot them. Crawl sources whose licence lets you use them.
"""
import argparse
import hashlib
import io
import re
import time
import urllib.request
import urllib.robotparser
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

UA = "heic-crawler/0.1 (research, honours robots.txt)"
MAX_PAGE = 5 << 20
MAX_IMG = 64 << 20
HEIC_RE = re.compile(r"\.(heic|heif)$", re.I)
PAGE_EXT = {"", ".html", ".htm", ".php", ".asp", ".aspx"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if not v:
                continue
            if k in ("href", "src", "data-src"):
                self.urls.append(v)
            elif k == "srcset":
                self.urls += [p.split()[0] for p in v.split(",") if p.strip()]


def iphone13_up(make, model):
    if "apple" not in (make or "").lower():
        return False
    model = model or ""
    m = re.match(r"iPhone (\d+)", model)
    return bool(m and int(m.group(1)) >= 13) or model.startswith("iPhone Air")


def exif_info(data):
    ex = Image.open(io.BytesIO(data)).getexif()
    taken = ex.get_ifd(0x8769).get(0x9003, "")
    return ex.get(0x010F, ""), ex.get(0x0110, ""), taken, bool(ex.get_ifd(0x8825))


def fetch(url, limit):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = r.headers.get("Content-Type", "")
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("too large")
    return ctype, data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", nargs="+")
    ap.add_argument("--out", default="heic_crawl")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--any-domain", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.tsv"
    seen_hash = {p.stem for p in out.glob("*.heic")}
    if not manifest.exists():
        manifest.write_text("url\tmodel\ttaken\thas_gps\tbytes\n")

    hosts = {urlparse(s).netloc for s in a.seeds}
    robots = {}

    def allowed(url):
        u = urlparse(url)
        if u.netloc not in robots:
            rp = urllib.robotparser.RobotFileParser(f"{u.scheme}://{u.netloc}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            robots[u.netloc] = rp
        rp = robots[u.netloc]
        return rp is None or rp.can_fetch(UA, url)

    queue = deque((s, 0) for s in a.seeds)
    visited, cands = set(a.seeds), []
    pages = kept = checked = 0

    def check(url):
        nonlocal kept, checked
        if not allowed(url):
            return
        time.sleep(a.delay)
        try:
            _, data = fetch(url, MAX_IMG)
            make, model, taken, gps = exif_info(data)
        except Exception as e:
            print(f"skip   {url}  ({e})")
            return
        checked += 1
        if not iphone13_up(make, model):
            print(f"drop   {url}  ({make} {model})".replace("  ( )", ""))
            return
        h = hashlib.sha1(data).hexdigest()
        if h in seen_hash:
            return
        seen_hash.add(h)
        (out / f"{h}.heic").write_bytes(data)
        with manifest.open("a") as f:
            f.write(f"{url}\t{model}\t{taken}\t{int(gps)}\t{len(data)}\n")
        kept += 1
        print(f"KEEP   {url}  ({model})")

    while queue and pages < a.max_pages:
        url, depth = queue.popleft()
        if HEIC_RE.search(urlparse(url).path):
            check(url)
            continue
        if not allowed(url):
            continue
        time.sleep(a.delay)
        try:
            ctype, data = fetch(url, MAX_PAGE)
        except Exception as e:
            print(f"skip   {url}  ({e})")
            continue
        pages += 1
        if "html" not in ctype:
            continue
        p = Links()
        p.feed(data.decode("utf-8", "replace"))
        for raw in p.urls:
            link = urldefrag(urljoin(url, raw))[0]
            u = urlparse(link)
            if u.scheme not in ("http", "https") or link in visited:
                continue
            if not a.any_domain and u.netloc not in hosts:
                continue
            ext = Path(u.path).suffix.lower()
            if HEIC_RE.search(u.path):
                visited.add(link)
                queue.appendleft((link, depth))  # check images first
            elif ext in PAGE_EXT and depth < a.max_depth:
                visited.add(link)
                queue.append((link, depth + 1))

    print(f"pages {pages}, heic checked {checked}, kept {kept} -> {out}")


if __name__ == "__main__":
    main()
