"""Validate static links, embedded media paths, and actual LFS media payloads."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

SITE = Path("site").resolve()
checked = set()

def check_url(url, page):
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path or "${" in url:
        return
    assert not parts.path.startswith("/"), f"Project-site root path: {url}"
    target = (page.parent / unquote(parts.path)).resolve()
    assert target.is_relative_to(SITE), f"Link leaves site: {page}: {url}"
    assert target.is_file(), f"Missing site asset: {page}: {url}"
    checked.add(target.relative_to(SITE).as_posix())

class Links(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.page = page
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src", "poster") and value:
                check_url(value, self.page)

def embedded(value, page):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("raw_video", "video", "poster", "provenance", "report", "trajectory") and isinstance(child, str):
                check_url(child, page)
            else:
                embedded(child, page)
    elif isinstance(value, list):
        for child in value:
            embedded(child, page)

manifest = []
for page in SITE.rglob("*.html"):
    source = page.read_text(encoding="utf-8")
    Links(page).feed(source)
    for attrs, content in re.findall(r"<script\b([^>]*)>(.*?)</script>", source, re.S):
        if "application/json" in attrs:
            embedded(json.loads(content), page)
for path in sorted(SITE.rglob("*")):
    if not path.is_file() or path.name == "site-manifest.json":
        continue
    with path.open("rb") as handle:
        first = handle.read(200)
    assert not first.startswith(b"version https://git-lfs.github.com/spec/v1"), f"Unresolved LFS pointer: {path}"
    if path.suffix == ".mp4":
        assert b"ftyp" in first[:32], f"Not an MP4 payload: {path}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    manifest.append({"path": path.relative_to(SITE).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
assert (SITE / "index.html").is_file()
result = {"passed": True, "checked_link_targets": len(checked), "files": manifest,
          "bytes": sum(item["bytes"] for item in manifest), "lfs_payloads_materialized": True}
(SITE / "site-manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "files"}))
