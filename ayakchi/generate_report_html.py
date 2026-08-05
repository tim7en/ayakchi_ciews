"""Render the CIEWS Markdown reference report as a self-contained styled HTML document."""

from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CIEWS_REFERENCE_REPORT.md"
TARGET = ROOT / "CIEWS_REFERENCE_REPORT.html"

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


def main():
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ayakchi CIEWS Technical Reference Report</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    TARGET.write_text(html, encoding="utf-8")
    print(f"Wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
