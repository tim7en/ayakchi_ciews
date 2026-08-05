"""Render the CIEWS Markdown reference report as a self-contained styled HTML document."""

import argparse
import base64
import mimetypes
from pathlib import Path
import re

import markdown


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "CIEWS_REFERENCE_REPORT.md"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; }
html { color: #1f2933; background: #eef2f3; font-family: Arial, Helvetica, sans-serif; }
body { max-width: 1020px; margin: 24px auto; padding: 54px 64px; background: white;
       box-shadow: 0 2px 14px rgba(0,0,0,.12); line-height: 1.48; }
h1 { color: #123f5a; font-size: 2.15rem; border-bottom: 4px solid #2a8f9d; padding-bottom: 12px; }
h2 { color: #155e75; margin-top: 2.2rem; border-bottom: 1px solid #b7d5da; padding-bottom: 6px; }
h3 { color: #1e6071; margin-top: 1.6rem; }
h4 { color: #365e66; }
p, li { font-size: 10.5pt; }
table { width: 100%; border-collapse: collapse; margin: 1.1rem 0 1.5rem; font-size: 9pt; }
th { background: #155e75; color: white; text-align: left; }
th, td { border: 1px solid #aebdc2; padding: 6px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f3f7f8; }
img { display: block; max-width: 100%; max-height: 820px; margin: 18px auto 8px; page-break-inside: avoid; }
blockquote { background: #e8f3f5; border-left: 5px solid #2a8f9d; margin: 1.2rem 0;
             padding: 8px 18px; color: #244b54; }
code { background: #eef2f3; padding: 1px 4px; }
pre { background: #f2f6f7; border: 1px solid #cad7da; padding: 12px; overflow-x: auto; font-size: 8.5pt; }
hr { border: 0; border-top: 1px solid #b7c7cb; margin: 2rem 0; }
strong { color: #173f4d; }
@media print {
  html { background: white; }
  body { max-width: none; margin: 0; padding: 0; box-shadow: none; }
  h1, h2, h3 { page-break-after: avoid; }
  table, figure, img, blockquote { page-break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""


def embed_local_images(html: str, source_dir: Path) -> str:
    """Convert relative local image links to data URIs for a portable report."""
    def replace(match):
        src = match.group(1)
        if src.startswith(("data:", "http://", "https://")):
            return match.group(0)
        image_path = (source_dir / src).resolve()
        if not image_path.is_file():
            return match.group(0)
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return match.group(0).replace(src, f"data:{mime};base64,{encoded}")
    return re.sub(r'<img\s+[^>]*src="([^"]+)"[^>]*>', replace, html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Ayakchi CIEWS Technical Reference Report")
    parser.add_argument("--embed-images", action="store_true")
    parser.add_argument("--presentation-reference", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.output.resolve() if args.output else source.with_suffix(".html")
    markdown_text = source.read_text(encoding="utf-8")
    if args.presentation_reference:
        markdown_text = markdown_text.replace(
            "## Source-aligned concept and evidence report",
            "## Final reference report for stakeholder presentation preparation",
            1,
        )
        markdown_text = re.sub(
            r"\*\*Status:\*\*[^\n]+",
            "**Status:** Consolidated presentation reference — institutional roles and technical thresholds remain for government confirmation  ",
            markdown_text,
            count=1,
        )
        marker = "---\n\n## Executive summary"
        request_note = """---

> **Response to the requested workshop preparation.** This report consolidates the Ayakchi technical evidence and the national-source-aligned CIEWS concept into one reference for preparing presentation materials. It clarifies the proposed CIEWS functions, intended users, water-resources applications, operational products, institutional questions, implementation pathway, and decisions requested from the Government of Uzbekistan. Maps and charts are screening evidence; they are not approved forecasts, warning thresholds, dam-safety findings, or engineering design outputs.

## Presentation proposition at a glance

| Question from the workshop request | Proposed answer |
|---|---|
| What is proposed? | A phased Ayakchi basin-and-reservoir information and action service connected to existing Uzhydromet, MoWR, MES/SEPRS, agricultural and local systems—not a standalone dashboard procurement. |
| Envisioned functions | Observe; quality-control and integrate data; interpret forecasts for Ayakchi; monitor basin and reservoir state; support operations and allocation; issue authorized warnings; communicate actions; learn after events. |
| Expected users | Uzhydromet, MoWR and dam operator, MES/SEPRS, MoA, WSS/WUA, regional and district authorities, infrastructure operators, communities and farmers. |
| Water-management applications | Reservoir operation, seasonal allocation, drought preparedness, irrigation scheduling, heavy-rainfall and rapid-runoff readiness, release notification, downstream protection, watershed and sediment management. |
| First government decision | Confirm pilot purpose, lead coordination, platform approach, product owners, warning authority, data-sharing commitments, initial users, and phased scope. |

## Recommended presentation narrative

1. Begin with the decisions Ayakchi users must make, not with technology.
2. Show the dual drought–flood operating problem and the importance of downstream tributaries.
3. Present CIEWS as an end-to-end service from forecast and observation to authorized action.
4. Demonstrate the proposed products and users, including the dashboard only as one product.
5. Separate available screening evidence from data, models and thresholds still required.
6. End with a short list of decisions for government confirmation and immediate assigned actions.

---

## Executive summary"""
        markdown_text = markdown_text.replace(marker, request_note, 1)
    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    if args.embed_images:
        body = embed_local_images(body, source.parent)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{args.title}</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    target.write_text(html, encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
