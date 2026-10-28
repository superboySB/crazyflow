"""Render the bid response as a self-contained local HTML preview.

All image, video and poster paths remain relative to the repository root. Static figures remain
visible when printing; animated GIFs and videos are hidden in print. Use --verify-browser to check
that Chrome can decode every image and read metadata and poster images for every embedded video.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
STYLE = """
:root {color-scheme:light;--ink:#182c42;--muted:#526273;--rule:#d8e0e9;--accent:#12558a;}
* {box-sizing:border-box;}
body {margin:0;background:#e9eef4;color:var(--ink);font:16px/1.85 "Noto Sans CJK SC",
"Microsoft YaHei","PingFang SC",sans-serif;}
main {max-width:1120px;margin:28px auto;padding:54px 66px 70px;background:white;
box-shadow:0 3px 24px #233b5118;}
h1 {font-size:29px;line-height:1.5;text-align:center;margin:0 0 30px;letter-spacing:.04em;}
h2 {font-size:22px;line-height:1.55;margin:38px 0 16px;padding:0 0 9px;
border-bottom:2px solid var(--accent);break-after:avoid;}
h3 {font-size:18px;line-height:1.6;margin:24px 0 10px;break-after:avoid;}
p {margin:12px 0;text-align:justify;overflow-wrap:anywhere;}
a {color:var(--accent);text-decoration:none;} a:hover {text-decoration:underline;}
blockquote {margin:20px 0;padding:8px 20px;border-left:4px solid var(--accent);background:#f2f6fa;}
blockquote p {margin:6px 0;}
img {display:block;max-width:100%;height:auto;margin:20px auto 10px;break-inside:avoid;}
video {display:block;width:100%;max-height:650px;object-fit:contain;margin:20px auto 10px;
background:#182c42;border:1px solid var(--rule);border-radius:3px;}
figure {margin:22px 0;break-inside:avoid;}
figcaption,.caption {font-size:14px;line-height:1.7;color:var(--muted);margin:8px 0 18px;}
table {border-collapse:collapse;width:100%;margin:20px 0;font-size:14px;line-height:1.7;}
th,td {border:1px solid var(--rule);padding:9px 11px;text-align:left;vertical-align:top;}
th {background:#edf3f9;font-weight:650;}
thead {display:table-header-group;}tr {break-inside:avoid;}
ul,ol {padding-left:1.8em;}li {margin:5px 0;}
code {font:13px/1.65 ui-monospace,"DejaVu Sans Mono",monospace;background:#edf2f6;
padding:2px 5px;border-radius:3px;}
pre {padding:16px;background:#f3f6f9;border:1px solid var(--rule);overflow:auto;line-height:1.6;}
pre code {padding:0;background:none;white-space:pre-wrap;word-break:break-word;}
hr {border:0;border-top:1px solid var(--rule);margin:30px 0;}
@media(max-width:760px){main{padding:25px 20px;margin:0;}h1{font-size:24px;}h2{font-size:20px;}}
@page {size:A4;margin:17mm 17mm 19mm;}
@media print {
body {background:white;font-size:10.5pt;line-height:1.75;}
main {max-width:none;padding:0;margin:0;box-shadow:none;}
h1 {font-size:20pt;}h2 {font-size:14pt;margin-top:22pt;}h3{font-size:12pt;}
video,img[src$=".gif" i],.motion,.motion-media,.motion-only {display:none!important;}
p:has(> img[src$=".gif" i]:only-child) {display:none;}
a {color:inherit;text-decoration:none;}
img {max-height:225mm;object-fit:contain;}
table {font-size:9pt;}pre {font-size:8pt;}
}
"""
CHECK_SCRIPT = r"""
<script>
window.addEventListener('load', async () => {
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const metadata = video => new Promise(resolve => {
    if (video.readyState >= 1) { resolve(); return; }
    video.addEventListener('loadedmetadata', resolve, {once:true});
    video.addEventListener('error', resolve, {once:true});
    video.load();
  });
  const videoList = [...document.querySelectorAll('video')];
  await Promise.race([Promise.all(videoList.map(metadata)), pause(6000)]);
  const images = [...document.images].map(im => ({src:im.getAttribute('src'),
    loaded:im.complete && im.naturalWidth>0, width:im.naturalWidth,height:im.naturalHeight}));
  const videos = await Promise.all([...document.querySelectorAll('video')].map(async video => {
    const poster = new Image(); poster.src = video.poster;
    await Promise.race([poster.decode().catch(()=>{}),pause(3000)]);
    return {src:video.getAttribute('src') || video.querySelector('source')?.getAttribute('src'),
      readyState:video.readyState,duration:Number.isFinite(video.duration)?video.duration:null,
      width:video.videoWidth,height:video.videoHeight,poster:video.getAttribute('poster'),
      posterLoaded:poster.complete && poster.naturalWidth>0,error:video.error?.message || null};
  }));
  const result={images,videos,ok:images.every(x=>x.loaded) && videos.every(x=>x.readyState>=1 &&
    x.duration>0 && x.width>0 && x.posterLoaded && !x.error)};
  const output=document.createElement('script');output.id='asset-validation';
  output.type='application/json';output.textContent=JSON.stringify(result);document.body.append(output);
});
</script>
"""


def local_path(value: str) -> Path:
    """Resolve a local media path, rejecting network and data dependencies."""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"Offline preview requires local media: {value}")
    resolved = (ROOT / unquote(parsed.path)).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError(f"Media file is absent or outside the repository: {value}")
    return resolved


class MediaHTML(HTMLParser):
    """Validate HTML media and add a local video poster when one can be inferred."""

    def __init__(self) -> None:
        """Initialize transformed markup and the preceding static figure."""
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.previous_static: str | None = None
        self.video_source: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Check media assets and write their opening tag."""
        values = dict(attrs)
        if tag == "img" and values.get("src"):
            local_path(values["src"])
            if not values["src"].lower().endswith(".gif"):
                self.previous_static = values["src"]
        if tag == "video":
            source = values.get("src")
            if source:
                local_path(source)
            if not values.get("poster"):
                poster = self.infer_poster(source) if source else self.previous_static
                if not poster:
                    raise ValueError("Each video needs a local poster or preceding static figure")
                attrs.append(("poster", poster))
                values["poster"] = poster
            local_path(values["poster"])
            if "preload" not in values:
                attrs.append(("preload", "metadata"))
            if "controls" not in values:
                attrs.append(("controls", None))
        if tag == "source" and values.get("src"):
            local_path(values["src"])
        attributes = "".join(
            f" {key}" if value is None else f' {key}="{html.escape(value, quote=True)}"'
            for key, value in attrs
        )
        self.parts.append(f"<{tag}{attributes}>")

    def infer_poster(self, source: str | None) -> str | None:
        """Prefer a same-name still, then a curated sequence, then the preceding still."""
        if not source:
            return self.previous_static
        path = local_path(source)
        candidates = [path.with_suffix(".png")]
        if path.name == "trajectory_tracking.mp4":
            candidates.append(path.with_name("tracking_sequence.png"))
        elif path.name in {"recovery_training.mp4", "deployment_excerpt.mp4"}:
            candidates.append(path.with_name("deployment_sequence.png"))
        candidates.extend(path.parent.glob("*sequence.png"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.relative_to(ROOT).as_posix()
        return self.previous_static

    def handle_endtag(self, tag: str) -> None:
        """Preserve an ending tag."""
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        """Preserve text data."""
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        """Preserve an entity reference."""
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        """Preserve a numeric character reference."""
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        """Preserve an HTML comment."""
        self.parts.append(f"<!--{data}-->")


def render(source: Path, destination: Path) -> None:
    """Convert Markdown to the local styled HTML preview."""
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    content = parser.render(source.read_text())
    media = MediaHTML()
    media.feed(content)
    document = (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(source.stem)}</title><style>{STYLE}</style></head>"
        f"<body><main>{''.join(media.parts)}</main>{CHECK_SCRIPT}</body></html>\n"
    )
    destination.write_text(document)


def verify_browser(destination: Path, report_path: Path) -> dict:
    """Use installed Chrome to verify that every local media item decodes."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise RuntimeError("Chrome or Chromium is required for --verify-browser")
    with tempfile.TemporaryDirectory(prefix="bid-preview-chrome-") as profile:
        result = subprocess.run(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--allow-file-access-from-files",
                "--no-first-run",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=12000",
                "--dump-dom",
                destination.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=35,
        )
    match = re.search(
        r'<script id="asset-validation" type="application/json">(.*?)</script>', result.stdout, re.S
    )
    if not match:
        raise RuntimeError(f"Chrome did not complete media validation: {result.stderr[-1500:]}")
    report = json.loads(match.group(1))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["ok"]:
        raise RuntimeError(f"Some preview media failed to load; see {report_path}")
    return report


def main() -> None:
    """Build the preview and optionally run browser asset verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "P标-RL.md")
    parser.add_argument("--output", type=Path, default=ROOT / "P标-RL.html")
    parser.add_argument("--verify-browser", action="store_true")
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
    print(f"Built {args.output}")
    if args.verify_browser:
        report = verify_browser(
            args.output.resolve(), ROOT / "peibiao_assets/preview-validation.json"
        )
        print(f"Chrome verified {len(report['images'])} images and {len(report['videos'])} videos")


if __name__ == "__main__":
    main()
